import React, { useState } from "react";
import { barChartSpec, DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, SectionNavigator, Slider, useDataApp } from "../../data-app-public.jsx";
import { MaterialFigure, TemporalFigure } from "./CaseFigures.jsx";
import { copy, sourcePreviews } from "./paper-copy.js";

const sections=[{id:"paper-abstract",label:"摘要"},{id:"paper-introduction",label:"研究问题"},{id:"paper-design",label:"数据与设计"},{id:"paper-methods",label:"方法"},{id:"paper-results",label:"结果"},{id:"paper-temporal",label:"时间信息"},{id:"paper-discussion",label:"讨论"},{id:"paper-conclusion",label:"结论"},{id:"paper-appendix",label:"术语与复现"},{id:"paper-references",label:"参考文献"}];
const similaritySpec={...barChartSpec({presentation:"plot",category:"pair",value:"cosine64",orientation:"horizontal",domain:[0,1],categoryWidth:100,labels:{value:true},format:{minimumFractionDigits:3,maximumFractionDigits:6}}),stackable:false,startAtZero:true,xLabel:"余弦相似度（无量纲）",valueDecimals:6};
const profileSpec={type:"line",x:"group",y:"文字查询",fields:["文字查询","封面","视频"],stackable:false,startAtZero:true,xLabel:"连续坐标组编号（每组32维）",yLabel:"组内RMS（无量纲）",valueDecimals:5};
const f=(value,digits=8)=>(Math.abs(Number(value))<.5*Math.pow(10,-digits)?0:Number(value)).toFixed(digits);

function Paragraphs({id,value,queries=[]}){
  const {snapshot,visible}=useDataApp();
  if(!visible(id))return null;
  const child=<RichNarrative id={id+":text"} value={value} sourcePreviews={sourcePreviews} label="编辑论文段落"/>;
  if(!queries.length)return <div id={id}>{child}</div>;
  const title=value.match(/^#+\s+(.+)/)?.[1]||value.match(/^\*\*(.+?)\*\*/)?.[1]||"方法与口径说明";
  return <ReportSection id={id} title={title} queryId={queries[0]} queryIds={queries} sourceRowsByQuery={Object.fromEntries(queries.map(q=>[q,snapshot.queries[q].rows]))} showHeading={false}>{child}</ReportSection>;
}
function Equations(){
  return <div className="paper-equations" aria-label="向量表示、归一化与余弦相似度定义">
    <div><span>z<sub>m</sub> = f<sub>θ</sub>(x<sub>m</sub>) ∈ ℝ<sup>768</sup>，m ∈ {"{文字，封面，视频}"}</span><span>(1)</span></div>
    <div><span>v = z / ‖z‖<sub>2</sub>；‖z‖<sub>2</sub> = √(∑<sub>j=1</sub><sup>768</sup> z<sub>j</sub><sup>2</sup>)</span><span>(2)</span></div>
    <div><span>s(q,v) = (q · v) / (‖q‖<sub>2</sub> ‖v‖<sub>2</sub>)</span><span>(3)</span></div>
    <div><span>t<sub>i</sub> ≈ k<sub>i</sub> / FPS<sub>nominal</sub></span><span>(4)</span></div>
  </div>;
}
function CoordinateInspector({rows}){
  const [dimension,setDimension]=useState(1);
  const row=rows.find(item=>item.dimension===dimension);
  return <div className="paper-coordinate-inspector" data-reviewed-rows contentEditable={false}>
    <Slider label="查看完整向量中的坐标维度" min={1} max={768} step={1} value={dimension} onChange={setDimension} showBounds formatValue={value=>"第 "+value+" 维"}/>
    <div className="paper-coordinate-values" aria-live="polite">{row&&[["文字查询",row.query],["封面",row.cover],["视频",row.visual]].map(([name,value])=><div key={name}><span>{name}</span><code>{f(value,9)}</code></div>)}</div>
  </div>;
}
export function ReportContent(){
  const {snapshot,visible,appTitle,canEdit,mode,setAppTitle}=useDataApp();
  const rows=id=>snapshot.queries[id].rows;
  const [caseRow]=rows("case"),frames=rows("frames");
  const table=(id,q,title,columns,caption)=>visible(id)&&<DataComponent id={id} kind="table" queryId={q} sourceRows={rows(q)} displayRows={rows(q)} title={title}><DataTable rows={rows(q)} columns={columns} searchable={false} compactNumbers={false} caption={caption}/></DataComponent>;
  return <article className="report-content paper-report" aria-label="多模态表示与视频时间信息方法研究报告">
    <header className="report-hero paper-hero">
      <RichNarrative id="paper:classification" className="paper-classification" value="技术报告 · 可复现单案例研究 · 2026年10月18日" label="编辑文献类型"/>
      <h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?event=>setAppTitle(event.currentTarget.textContent.trim()||appTitle):undefined} onKeyDown={canEdit&&mode==="edit"?event=>{if(event.key==="Enter"){event.preventDefault();event.currentTarget.blur();}}:undefined}>{appTitle}</h1>
      <RichNarrative id="paper:subtitle" className="paper-subtitle" value={"EmbeddingGemma 2 的可复现单样本案例研究\n\nCross-modal representations and temporal information in video retrieval"} label="编辑副标题"/>
    </header>
    <SectionNavigator sections={sections} placement="auto"/>
    <Paragraphs id="paper-abstract" value={copy.abstract} queries={["case","similarity"]}/>
    <Paragraphs id="paper-introduction" value={copy.introduction}/>
    <Paragraphs id="paper-design" value={copy.design} queries={["case","frames"]}/>
    {visible("figure-material")&&<DataComponent id="figure-material" kind="custom" queryId="case" queryIds={["case","frames"]} sourceRowsByQuery={{case:rows("case"),frames}} displayRows={rows("case")} title="图1 · 同一素材的三种输入与编码接口"><MaterialFigure caseRow={caseRow} frames={frames}/></DataComponent>}
    <Paragraphs id="paper-material-caption" value={copy.materialCaption} queries={["case","frames"]}/>
    <Paragraphs id="paper-methods" value={copy.methods} queries={["case","coordinates","frames"]}/>
    <Equations/>
    <Paragraphs id="paper-equation-notes" value={copy.equationNotes} queries={["coordinates","similarity","frames"]}/>
    <Paragraphs id="paper-results" value={copy.results} queries={["statistics","similarity"]}/>
    {table("table-vector-statistics","statistics","表1 · 三条固定向量的描述性统计",[{field:"kind",label:"输入"},{field:"dimension",label:"维数"},{field:"l2_norm",label:"L2范数",renderCell:value=>f(value,9)},{field:"rms",label:"整体RMS",renderCell:value=>f(value,8)},{field:"raw_bytes",label:"原始字节",renderCell:value=>String(value)}],"三种输入向量的维度、范数、RMS和原始存储量")}
    <Paragraphs id="paper-statistics-note" value={copy.statisticsNote} queries={["statistics"]}/>
    {visible("figure-similarity")&&<EvidenceChart id="figure-similarity" queryId="similarity" title="图2 · 三对固定向量的余弦相似度" spec={similaritySpec} rows={rows("similarity")} sourceRows={rows("similarity")} height={250}/>}
    <Paragraphs id="paper-similarity-caption" value={copy.similarityCaption} queries={["similarity","case"]}/>
    {table("table-pair-numerics","similarity","表2 · 独立复算与数值一致性",[{field:"pair",label:"向量配对"},{field:"cosine64",label:"float64余弦",renderCell:value=>f(value,9)},{field:"dot32",label:"float32点积",renderCell:value=>f(value,9)},{field:"absolute_error",label:"绝对差",renderCell:value=>Number(value).toExponential(2)},{field:"l2",label:"L2距离",renderCell:value=>f(value,6)}],"三对余弦与点积的独立数值核验")}
    {visible("figure-coordinate-profile")&&<EvidenceChart id="figure-coordinate-profile" queryId="dimension_bins" title="图3 · 连续坐标分组的数值尺度" spec={profileSpec} rows={rows("dimension_bins")} sourceRows={rows("dimension_bins")} height={310}/>}
    <Paragraphs id="paper-profile-caption" value={copy.profileCaption} queries={["dimension_bins","coordinates"]}/>
    <Paragraphs id="paper-temporal" value={copy.temporal} queries={["frames","case"]}/>
    {visible("figure-time")&&<DataComponent id="figure-time" kind="custom" queryId="frames" queryIds={["frames","case"]} sourceRowsByQuery={{frames,case:rows("case")}} displayRows={frames} title="图4 · 固定画面输入下的时间伸缩思想实验"><TemporalFigure frames={frames} duration={caseRow.video_duration_s}/></DataComponent>}
    <Paragraphs id="paper-time-caption" value={copy.timeCaption} queries={["frames","case"]}/>
    {table("table-frame-times","frames","表3 · 采样帧与名义时间的对应关系",[{field:"index",label:"采样序号",renderCell:value=>String(value+1)},{field:"source_frame_index",label:"原帧号（0起）"},{field:"timestamp_s",label:"名义时间 / s",renderCell:value=>f(value,4)},{field:"decoder_delta_ms",label:"解码位置差 / ms",renderCell:value=>f(value,3)}],"实际采样帧、名义时间与解码器位置差")}
    <Paragraphs id="paper-discussion" value={copy.discussion} queries={["case","frames","similarity"]}/>
    <Paragraphs id="paper-conclusion" value={copy.conclusion} queries={["case","similarity"]}/>
    <Paragraphs id="paper-appendix" value={copy.glossary}/>
    <Paragraphs id="paper-reproducibility" value={copy.reproducibility} queries={["case","coordinates","frames","similarity"]}/>
    {visible("coordinate-inspector")&&<DataComponent id="coordinate-inspector" kind="custom" queryId="coordinates" sourceRows={rows("coordinates")} displayRows={rows("coordinates")} title="附录B · 完整768维坐标检查"><CoordinateInspector rows={rows("coordinates")}/></DataComponent>}
    <Paragraphs id="paper-references" value={copy.references}/>
    <RichNarrative id="paper:footer" className="paper-footer" label="编辑页尾" value="文献类型：本地技术报告。本文未发表、未经过同行评审；未声明作者机构或虚构 DOI。原始素材仅用于当前本地案例说明。"/>
  </article>;
}
