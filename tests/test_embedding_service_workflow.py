"""Workflow order and stop-on-failure, no live API or credentials."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

SPEC=importlib.util.spec_from_file_location("embedding_workflow",Path(__file__).parents[1]/"scripts/embedding_service.py")
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class WorkflowTests(unittest.TestCase):
    def config(self):
        return {"config_file":Path("/synthetic/control.json"),"client_env_file":Path("/synthetic/client.env"),"endpoint_file":Path("/synthetic/endpoint.json")}

    def test_start_orders_control_before_readiness_then_optional_test(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            result={"ok":True,"data":{}} if command[1].endswith("manager.py") else {"status":"pass","boot_id":"synthetic"}
            return SimpleNamespace(returncode=0,stdout=json.dumps(result))
        flow=module.Workflow(self.config(),executor=execute,clock=lambda:0)
        flow.pin_configuration=lambda:None
        published=[]
        flow.publish_verified_configuration=lambda ready:published.append(ready)
        result=flow.run("start",test_after_start=True)
        self.assertEqual(result["status"],"pass")
        self.assertEqual([s["step"] for s in result["steps"]],["power-on","refresh-endpoint","wait-ready","test"])
        self.assertEqual(len(published),1)
        self.assertTrue(all(c[0]==module.sys.executable for c in calls))
        self.assertFalse(result["automatic_shutdown"])
        self.assertFalse(result["automatic_new_instance"])

    def test_failed_power_on_does_not_poll_probe_or_repeat_write(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=2,stdout=json.dumps({"ok":False,"error":"capacity_unavailable","outcome_unknown":False}))
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("start")
        self.assertEqual(result["status"],"fail")
        self.assertEqual(len(calls),1)
        self.assertEqual(result["error"],"power-on_failed")

    def test_readiness_failure_does_not_publish_configuration(self):
        def execute(command,**kwargs):
            manager=command[1].endswith("manager.py")
            return SimpleNamespace(returncode=0 if manager else 1,stdout=json.dumps({"ok":True,"data":{}} if manager else {"status":"fail","error":"not_ready"}))
        flow=module.Workflow(self.config(),executor=execute,clock=lambda:0)
        flow.pin_configuration=lambda:None
        flow.publish_verified_configuration=lambda ready:self.fail("must not publish before ready")
        result=flow.run("start")
        self.assertEqual(result["status"],"fail")
        self.assertEqual(result["error"],"wait-ready_failed")

    def test_write_timeout_is_unknown_and_not_retried(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            raise subprocess.TimeoutExpired(command,1)
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("start")
        self.assertTrue(result["outcome_unknown"])
        self.assertEqual(len(calls),1)

    def test_test_command_needs_no_account_management(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0,stdout='{"status":"pass"}')
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("test")
        self.assertEqual(result["status"],"pass")
        self.assertEqual(len(calls),1)
        self.assertTrue(calls[0][1].endswith("probe.py"))

    def test_invalid_child_stdout_is_not_echoed(self):
        def execute(command,**kwargs):
            return SimpleNamespace(returncode=1,stdout="sensitive-untrusted-output",stderr="secret")
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("images")
        self.assertNotIn("sensitive",json.dumps(result))
        self.assertNotIn("secret",json.dumps(result))

    def test_missing_image_name_stops_before_network(self):
        def execute(*args,**kwargs):
            self.fail("no API without name")
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("image-save")
        self.assertEqual(result["error"],"image_name_required")

    def test_timeout_rejects_invalid_budget(self):
        for value in [0,-1,float("nan"),float("inf"),3601,True]:
            with self.subTest(value=value):
                with self.assertRaises(module.WorkflowError):
                    module.Workflow(self.config(),timeout=value)

    def test_accepted_image_id_survives_wait_timeout(self):
        calls=[]
        def execute(command,**kwargs):
            calls.append(command)
            if len(calls)==1:
                return SimpleNamespace(returncode=0,stdout='{"ok":true,"data":{"image_uuid":"image-synthetic123"}}')
            raise subprocess.TimeoutExpired(command,1)
        result=module.Workflow(self.config(),executor=execute,clock=lambda:0).run("image-save",name="synthetic")
        self.assertEqual(result["status"],"fail")
        self.assertEqual(result["accepted_image_uuid"],"image-synthetic123")
        self.assertIn("image-wait",result["resume_command"])
        self.assertNotIn("--wait",calls[0])
        self.assertEqual(len(calls),2)

    def test_real_configuration_sync_and_changed_url_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=root/"client.env";endpoint=root/"endpoint.json"
            env.write_text('export OMNI_INFERENCE_URL=https://old.example.com\nexport OMNI_INFERENCE_API_KEY=synthetic-test-key-1234567890\n')
            env.chmod(0o600)
            endpoint.write_text('{"url":"https://new.example.com","status":"discovered"}')
            config={"config_file":root/"control.json","client_env_file":env,"endpoint_file":endpoint}
            flow=module.Workflow(config,clock=lambda:0)
            flow.pin_configuration()
            flow.publish_verified_configuration({"boot_id":"synthetic-boot"})
            self.assertIn('https://new.example.com',env.read_text())
            self.assertEqual(env.stat().st_mode&0o777,0o600)
            value=json.loads(endpoint.read_text())
            self.assertEqual(value['status'],'ready')
            self.assertEqual(value['boot_id'],'synthetic-boot')
            endpoint.write_text('{"url":"https://changed.example.com"}')
            with self.assertRaises(module.WorkflowError):
                flow.publish_verified_configuration({"boot_id":"synthetic"})
            flow._temporary.cleanup()


if __name__=="__main__":
    unittest.main()
