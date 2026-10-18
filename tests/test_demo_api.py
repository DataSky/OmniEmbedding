"""HTTP integration boundaries for the isolated local demo (no real encoder)."""
from pathlib import Path
import io

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from omniembed.demo_app import create_app


class FakeEngine:
    def __init__(self, directory):
        self.directory = directory
        self.started = self.stopped = False
        self.cover = directory / 'cover.jpg'
        Image.new('RGB', (32, 18), 'blue').save(self.cover)
        self.video = directory / 'video.mp4'
        self.video.write_bytes(b'x' * 256)

    def start(self): self.started = True
    def stop(self): self.stopped = True
    def overview(self): return {'assets': 1, 'storage': {'backend': 'SQLite'}}
    def datasets(self): return [{'id': 'd1'}]
    def preview(self, path): return {'total_assets': 1, 'path': path}
    def submit_import(self, path, **kwargs): return {'id': 'j1', 'status': 'queued'}
    def submit_embed(self, **kwargs): return {'id': 'j2', 'status': 'queued'}
    def jobs(self): return [{'id': 'j1'}]
    def job(self, ident): return {'id': ident} if ident == 'j1' else None
    def cancel(self, ident): return {'id': ident, 'cancel_requested': True}
    def assets(self, **kwargs): return {'items': [self.asset('a1')], 'total': 1}
    def asset(self, ident):
        if ident == 'missing': return None
        return {'id': ident, 'title': '测试', 'cover_path': str(self.cover if ident != 'outside' else self.directory.parent / 'private.jpg'),
                'video_path': str(self.video), 'frames': [{'index': 0, 'path': str(self.cover)}]}
    def search(self, query, **kwargs): return {'trace_id': 't1', 'items': [], 'mode': kwargs['mode']}
    def search_image(self, image, **kwargs): return {'trace_id': 't2', 'items': [], 'image_size': list(image.size)}
    def galaxy(self, **kwargs): return {'nodes': [], 'edges': [], 'projection': {'method': 'PCA'}}
    def traces(self): return [{'id': 't1'}]
    def trace(self, ident): return {'id': ident} if ident == 't1' else None
    def health(self, probe=False): return {'ready': True, 'probe': probe}


@pytest.fixture
def demo_client(tmp_path):
    directory = tmp_path / 'data'
    directory.mkdir()
    Image.new('RGB', (8, 8), 'red').save(tmp_path / 'private.jpg')
    engine = FakeEngine(directory)
    app = create_app(data_dir=directory, engine=engine)
    with TestClient(app, base_url='http://127.0.0.1:8780') as client:
        yield client, engine
    assert engine.started and engine.stopped


def test_loopback_host_and_origin_guards(demo_client):
    client, _ = demo_client
    assert client.get('/api/demo/overview').status_code == 200
    assert client.get('/api/demo/overview', headers={'Host': 'attacker.example'}).status_code == 403
    assert client.post('/api/demo/import/preview', json={'path': '/tmp/source.tar'}, headers={'Origin': 'https://attacker.example'}).status_code == 403
    assert client.post('/api/demo/import/preview', json={'path': '/tmp/source.tar'}, headers={'Origin': 'http://127.0.0.1:8780'}).status_code == 200


def test_registered_media_and_range_without_arbitrary_files(demo_client):
    client, _ = demo_client
    assert client.get('/api/demo/media/a1/cover').headers['content-type'].startswith('image/jpeg')
    assert client.get('/api/demo/media/a1/frame/0').status_code == 200
    assert client.get('/api/demo/media/a1/frame/55').status_code == 404
    assert client.get('/api/demo/media/outside/cover').status_code == 404
    assert client.get('/api/demo/media/missing/video').status_code == 404
    result = client.get('/api/demo/media/a1/video', headers={'Range': 'bytes=0-15'})
    assert result.status_code == 206 and len(result.content) == 16
    assert client.get('/data/autodl/inference.key').status_code == 404


def test_search_input_validation_and_real_image_decode(demo_client):
    client, _ = demo_client
    assert client.post('/api/demo/search', json={'query': '做饭', 'mode': 'cover'}).json()['mode'] == 'cover'
    assert client.post('/api/demo/search', json={'query': 'q', 'mode': 'invalid'}).status_code == 422
    assert client.post('/api/demo/search', json={'query': 'q', 'model': 'unexpected'}).status_code == 422
    assert client.post('/api/demo/search', json={'query': 'q', 'top_k': 1000}).status_code == 422
    assert client.post('/api/demo/search/image', files={'file': ('x.png', b'bad', 'image/png')}).status_code == 400
    stream = io.BytesIO(); Image.new('RGB', (8, 6), 'blue').save(stream, format='PNG')
    response = client.post('/api/demo/search/image', data={'mode': 'visual'}, files={'file': ('query.png', stream.getvalue(), 'image/png')})
    assert response.status_code == 200 and response.json()['image_size'] == [8, 6]
    # Camera JPEG orientation must match the cover ingestion path.
    camera = Image.new('RGB', (8, 6), 'blue')
    orientation = Image.Exif(); orientation[274] = 6
    stream = io.BytesIO(); camera.save(stream, format='JPEG', exif=orientation)
    response = client.post('/api/demo/search/image', data={'mode': 'cover'}, files={'file': ('camera.jpg', stream.getvalue(), 'image/jpeg')})
    assert response.status_code == 200 and response.json()['image_size'] == [6, 8]


def test_request_limits_apply_before_body_parsing(demo_client):
    client, _ = demo_client
    assert client.post('/api/demo/import/preview', content=b'x', headers={'Content-Length': str(513 * 1024)}).status_code == 413
    assert client.post('/api/demo/search/image', content=b'x', headers={'Content-Length': str(13 * 1024**2)}).status_code == 413
    assert client.post('/api/demo/search/image', content=b'x', headers={'Content-Length': '-1'}).status_code == 400
    def chunks():
        yield b'x' * (256 * 1024)
        yield b'x' * (257 * 1024)
    assert client.post('/api/demo/import/preview', content=chunks()).status_code == 413


def test_lifecycle_and_not_found_contracts(demo_client):
    client, engine = demo_client
    assert engine.started
    assert client.post('/api/demo/import', json={'path': '/tmp/source.tar', 'auto_embed': False}).status_code == 202
    assert client.post('/api/demo/jobs', json={'kind': 'embed'}).status_code == 202
    assert client.get('/api/demo/jobs/missing').status_code == 404
    assert client.get('/api/demo/assets/missing').status_code == 404
    assert client.get('/api/demo/traces/missing').status_code == 404
    assert client.post('/api/demo/health/probe').json()['probe']
