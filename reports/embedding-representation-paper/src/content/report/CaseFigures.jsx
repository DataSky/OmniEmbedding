import React, { useState } from "react";
import { Slider } from "../../data-app-public.jsx";
import media from "../assets/example-media.json";
import "./case-figures.css";

const seconds = (value) => `${Number(value).toFixed(2)} s`;
const frameImage = (index) => media.frames.find((frame) => frame.index === index)?.data_url;

function CaseImage({ src, alt, ...props }) {
  return src ? <img src={src} alt={alt} {...props} /> : <div className="paper-private-media" role="img" aria-label={`${alt}；私有素材，未随源码公开`}><span>私有素材，未随源码公开</span></div>;
}

export function MaterialFigure({ caseRow = {}, frames = [] }) {
  const ordered = [...frames].sort((a, b) => a.index - b.index);
  const dimension = caseRow.dimension ?? 768;
  const routes = [
    { kind: "文字", input: "一句查询", process: "文字预处理与编码" },
    { kind: "图片", input: "一张封面", process: "图像预处理与编码" },
    { kind: "视频", input: `${ordered.length} 张有序采样帧`, process: "有序画面联合编码" },
  ];
  return <div className="paper-figure paper-figure--material" data-reviewed-rows contentEditable={false}>
    <div className="paper-figure-inputs">
      <div className="paper-figure-cover">
        <CaseImage src={media.cover.data_url} alt="案例视频的真实封面" width={caseRow.cover_width} height={caseRow.cover_height} />
        <span>同一素材的封面</span>
      </div>
      <div className="paper-figure-query">
        <span className="paper-figure-label">文字查询 · 与封面属于不同输入</span>
        <blockquote>{caseRow.query}</blockquote>
        <dl className="paper-figure-facts">
          <div><dt>独立视频样本</dt><dd>{caseRow.independent_case_n ?? 1}</dd></div>
          <div><dt>原始帧数</dt><dd>{caseRow.original_frames}</dd></div>
          <div><dt>抽取帧数</dt><dd>{ordered.length}</dd></div>
          <div><dt>名义时长</dt><dd>{seconds(caseRow.video_duration_s)}</dd></div>
        </dl>
        <p className="paper-figure-note">公开别名为sample-001，原始平台标识未公开。私有封面与采样帧未随源码公开，帧号与时间保留实测值。查询句用于说明跨模态比较，并非视频字幕或人工相关性标签。</p>
      </div>
    </div>

    <div className="paper-figure-film-heading"><span>实际送入视频编码的有序画面</span><span>帧号从 0 开始 · 时间为近似值</span></div>
    <div className="paper-figure-filmstrip">
      {ordered.map((frame) => <div className="paper-figure-frame" key={frame.index}>
        <CaseImage src={frameImage(frame.index)} alt={`第 ${frame.index + 1} 张实际采样图，原视频帧号 ${frame.source_frame_index}`} width={frame.width} height={frame.height} />
        <span className="paper-figure-frame-number">{String(frame.index + 1).padStart(2, "0")}</span>
        <div className="paper-figure-frame-info"><strong>{seconds(frame.timestamp_s)}</strong><span>原帧 {frame.source_frame_index}</span></div>
      </div>)}
    </div>

    <div className="paper-figure-flow" aria-label="三种输入通过各自处理得到同一维数的向量">
      {routes.map((route) => <div className="paper-figure-flow-row" key={route.kind}>
        <div className="paper-figure-flow-input"><span>{route.kind}</span><strong>{route.input}</strong></div>
        <span className="paper-figure-arrow" aria-hidden="true">→</span>
        <div className="paper-figure-flow-process">{route.process}</div>
        <span className="paper-figure-arrow" aria-hidden="true">→</span>
        <div className="paper-figure-flow-output"><strong>{dimension}</strong><span>维向量</span></div>
      </div>)}
    </div>
    <p className="paper-figure-note paper-figure-note--ruled">流程图呈现输入与输出的接口关系。内部注意力、特征汇总及其因果贡献未在本例中直接观测；三条向量的维数相同，但数值不同。</p>
  </div>;
}

function TimelineTrack({ title, duration, frames, multiplier = 1, hypothesis = false }) {
  return <div className={`paper-figure-track${hypothesis ? " paper-figure-track--hypothesis" : ""}`}>
    <div className="paper-figure-track-heading"><strong>{title}</strong><span>总时长 {seconds(duration * multiplier)}</span></div>
    <div className="paper-figure-track-line" aria-hidden="true">
      {frames.map((frame) => <span key={frame.index} style={{ left: `${Math.max(0, Math.min(100, frame.timestamp_s / duration * 100))}%` }}>{frame.index + 1}</span>)}
    </div>
    <div className="paper-figure-timestamps">
      {frames.map((frame) => <div key={frame.index}><span>帧 {frame.index + 1}</span><strong>{seconds(frame.timestamp_s * multiplier)}</strong></div>)}
    </div>
  </div>;
}

export function TemporalFigure({ frames = [], duration }) {
  const [multiplier, setMultiplier] = useState(5);
  const ordered = [...frames].sort((a, b) => a.index - b.index);
  const safeDuration = Number(duration) > 0 ? Number(duration) : 1;
  return <div className="paper-figure paper-figure--temporal" data-reviewed-rows contentEditable={false}>
    <div className="paper-figure-experiment-heading"><span className="paper-figure-label">思想实验 · 未重新执行模型推理</span><span className="paper-figure-fixed">{ordered.length} 张图像及先后顺序保持相同</span></div>
    <div className="paper-figure-time-control">
      <Slider id="paper-time-multiplier" label="假设时间轴拉伸倍率" min={1} max={10} step={1}
        value={multiplier} onChange={setMultiplier} showBounds formatValue={(value) => `${value}×`} />
    </div>
    <TimelineTrack title="A · 真实素材的名义时间轴" duration={safeDuration} frames={ordered} />
    <TimelineTrack title="B · 假设拉伸后的时间轴" duration={safeDuration} frames={ordered} multiplier={multiplier} hypothesis />
    <div className="paper-figure-time-summary" aria-live="polite">
      <div><span>像素序列</span><strong>完全相同</strong><small>F₁ → F₂ → … → F₈</small></div>
      <span className="paper-figure-time-divider" aria-hidden="true">≠</span>
      <div><span>假设实际用时</span><strong>{seconds(safeDuration)} → {seconds(safeDuration * multiplier)}</strong><small>仅由倍率计算，并非第二段实测视频</small></div>
    </div>
    <p className="paper-figure-note paper-figure-note--ruled">当前接口只传送有序画面，未传入秒数、FPS 或音轨。因此，时间安排发生变化而画面输入不变时，编码器没有新增的时间字段可用于区分两者。此处不报告新向量或时间辨别准确率。</p>
  </div>;
}
