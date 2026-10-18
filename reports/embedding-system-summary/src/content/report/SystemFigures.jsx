import React from "react";
import media from "../assets/example-media.json";

function Box({label,detail}){return <div className="system-node"><strong>{label}</strong><span>{detail}</span></div>;}

function CaseImage({src,alt,...props}){
  return src ? <img src={src} alt={alt} {...props}/> : <div className="system-private-media" role="img" aria-label={`${alt}；私有素材，未随源码公开`}><span>私有素材，未随源码公开</span></div>;
}

export function ArchitectureFigure(){
  return <div className="system-architecture" data-reviewed-rows contentEditable={false}>
    <div className="system-architecture-client"><Box label="浏览器 · 本机8780" detail="素材库 / 搜索 / Galaxy / 流水线 / 存储与Trace"/><span className="system-connector">↓ 本机HTTP</span></div>
    <div className="system-architecture-planes">
      <div className="system-plane system-plane-local"><span className="system-plane-label">本机业务面 · 保存素材与索引</span>
        <Box label="FastAPI + DemoEngine" detail="输入校验 · 单任务worker · 本进程编码锁"/>
        <div className="system-two-nodes"><Box label="媒体与预处理" detail="受管拷贝 / SHA / 采样帧"/><Box label="检索与Galaxy" detail="NumPy / BM25 / RRF / PCA"/></div>
        <Box label="SQLite WAL" detail="8张表：素材、向量、任务、阶段与历史证据"/>
        <Box label="RemoteEncoder · 本机后端" detail="携独立Key发HTTPS；验证返回模型身份和坐标"/>
      </div>
      <div className="system-plane system-plane-cloud"><span className="system-plane-label">AutoDL数据面 · 仅模型编码</span>
        <Box label="HTTPS推理入口" detail="Bearer鉴权 / 输入类型与大小校验"/>
        <span className="system-connector">↓ 文字 / 图片 / 1–8张有序帧</span>
        <Box label="EmbeddingGemma 2 · RTX 5090" detail="固定revision · BF16 · 1个共享推理槽"/>
        <span className="system-connector">↓ 每个输入一条768维表示</span>
        <Box label="返回本机" detail="float32坐标 / 版本 / boot_id / 校验结果"/>
      </div>
    </div>
    <div className="system-architecture-control"><strong>AutoDL控制面</strong><span>开发者Token → 资源 / 实例 / 镜像管理；与推理Bearer Key分离</span></div>
  </div>;
}

export function FlowFigure({rows}){
  return <ol className="system-flow" data-reviewed-rows contentEditable={false} aria-live="polite">
    {rows.map(row=><li key={row.flow+row.step}><span className="system-flow-number">{String(row.step).padStart(2,"0")}</span><strong>{row.name}</strong><p>{row.operation}</p><code>{row.evidence}</code></li>)}
  </ol>;
}

export function CaseFigure({caseRow,frames}){
  return <div className="system-case" data-reviewed-rows contentEditable={false}>
    <div className="system-case-intro"><CaseImage src={media.cover.data_url} alt="案例素材的真实封面" width="360" height="640"/><div><span className="system-plane-label">查询与真实素材</span><blockquote>{caseRow.query}</blockquote><dl><div><dt>独立案例</dt><dd>1条事后Top1素材</dd></div><div><dt>原视频</dt><dd>{caseRow.original_frames}帧 / {Number(caseRow.video_duration_s).toFixed(2)}秒</dd></div><div><dt>编码输入</dt><dd>{frames.length}张有序画面</dd></div></dl><p>公开别名为sample-001，原始平台标识未公开。私有封面与采样帧未随源码公开，以下保留实测帧号与时间；未独立标注相关性。名义时间由帧号/FPS估计。</p></div></div>
    <div className="system-filmstrip">{frames.map(frame=><figure key={frame.index}><CaseImage src={media.frames.find(f=>f.index===frame.index)?.data_url} alt={`第${frame.index+1}张实际采样帧，原帧号${frame.source_frame_index}`} width={frame.width} height={frame.height}/><figcaption><strong>{Number(frame.timestamp_s).toFixed(2)} s</strong><span>原帧{frame.source_frame_index}</span></figcaption></figure>)}</div>
  </div>;
}
