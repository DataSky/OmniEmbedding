import React, { useState } from "react";
import { barChartSpec, DataComponent, DataTable, Dropdown, EvidenceChart, ReportSection, RichNarrative, SectionNavigator, useDataApp } from "../../data-app-public.jsx";
import copy from "./system-copy.json";
import { ArchitectureFigure, CaseFigure, FlowFigure } from "./SystemFigures.jsx";
import "./system-figures.css";

const sourcePreviews={
  "https://huggingface.co/docs/transformers/en/model_doc/embedding_gemma2":{title:"EmbeddingGemma2 · Transformers",source:"Hugging Face 官方文档",summary:"说明共享768维表示、任务提示词、较短前缀向量及归一化。当前调用与时间处理仍以本地实现为准。",approvedForReport:true},
  "https://www.autodl.com/docs/common_api/":{title:"AutoDL 通用API",source:"AutoDL 官方文档",summary:"说明开发者Token和账户控制面。独立推理Bearer Key与账户Token用途不同，实际服务接口以本地代码为准。",approvedForReport:true}
};
const navigation=[
  ["summary","阅读摘要"],["scope","工作全貌"],["architecture","总体架构"],["concepts","关键概念"],
  ["deployment","部署与保存"],["api","服务调用"],["data","素材导入"],["encoding","索引与存储"],
  ["retrieval","召回"],["case","真实案例"],["time","时间与Galaxy"],["whitebox","白盒追踪"],
  ["quality","质量评测"],["performance","并发与稳定性"],["operations","运维恢复"],["next","后续建议"],["appendix","交付与证据"]
].map(([id,label])=>({id:"system-"+id,label}));
const throughputSpec={...barChartSpec({presentation:"plot",category:"concurrency",value:"success_rps",orientation:"horizontal",domain:[0,20],categoryWidth:68,labels:{value:true},format:{minimumFractionDigits:2,maximumFractionDigits:2}}),stackable:false,startAtZero:true,xLabel:"通过向量校验的成功请求 / 秒",valueDecimals:2};
const timingSpec={...barChartSpec({presentation:"plot",category:"stage",value:"elapsed_ms",orientation:"horizontal",domain:[0,320],categoryWidth:130,labels:{value:true},format:{minimumFractionDigits:2,maximumFractionDigits:2}}),stackable:false,startAtZero:true,xLabel:"已有单次请求的分阶段耗时 / ms",valueDecimals:2};
const number=(v,d=2)=>Number(v).toFixed(d);

function Narrative({name,queries=[]}){
  const {snapshot,visible}=useDataApp();
  const id="system-"+name;
  if(!visible(id))return null;
  const child=<RichNarrative id={id+":body"} value={copy[name]} label="编辑系统总结段落" sourcePreviews={sourcePreviews}/>;
  if(!queries.length)return <section id={id}>{child}</section>;
  const title=copy[name].match(/^#+\s+(.+)/)?.[1]||"说明";
  return <ReportSection id={id} title={title} queryId={queries[0]} queryIds={queries} sourceRowsByQuery={Object.fromEntries(queries.map(q=>[q,snapshot.queries[q].rows]))} showHeading={false}>{child}</ReportSection>;
}

function FlowExplorer(){
  const {snapshot,visible}=useDataApp();
  const [flow,setFlow]=useState("ingest");
  const rows=snapshot.queries.flows.rows.filter(r=>r.flow===flow);
  if(!visible("figure-flow"))return null;
  return <DataComponent id="figure-flow" kind="custom" queryId="flows" sourceRows={rows} displayRows={rows} title="图2 · 切换查看三个可追溯流程" headerControls={<Dropdown label="查看流程" showLabel value={flow} choices={["ingest","search","audit"]} choiceLabels={{ingest:"导入与构建索引",search:"一次在线检索",audit:"从结果反查证据"}} onChange={setFlow}/>}><FlowFigure rows={rows}/></DataComponent>;
}

export function ReportContent(){
  const {snapshot,visible,appTitle,canEdit,mode,setAppTitle}=useDataApp();
  const rows=q=>snapshot.queries[q].rows;
  const table=(id,q,title,columns,caption)=>visible(id)&&<DataComponent id={id} kind="table" queryId={q} sourceRows={rows(q)} displayRows={rows(q)} title={title}><DataTable rows={rows(q)} columns={columns} searchable={false} compactNumbers={false} caption={caption}/></DataComponent>;
  return <article className="report-content system-report" aria-label="OmniEmbedding 部署与检索系统总结报告">
    <header className="report-hero system-hero">
      <RichNarrative id="system:classification" className="system-classification" value="系统工程总结 · 2026年10月18日" label="编辑报告标识"/>
      <h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit"?e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined} onKeyDown={canEdit&&mode==="edit"?e=>{if(e.key==="Enter"){e.preventDefault();e.currentTarget.blur();}}:undefined}>{appTitle}</h1>
      <RichNarrative id="system:subtitle" className="report-deck" value="模型部署与调用 · 素材导入与向量索引 · 召回、Galaxy和全链路证据" label="编辑副标题"/>
    </header>
    <SectionNavigator sections={navigation} placement="auto"/>
    <Narrative name="summary" queries={["inventory","soak","deployment"]}/>
    <Narrative name="scope"/>
    <Narrative name="architecture" queries={["architecture"]}/>
    {visible("figure-architecture")&&<DataComponent id="figure-architecture" kind="custom" queryId="architecture" sourceRows={rows("architecture")} displayRows={rows("architecture")} title="图1 · 本机业务与远程GPU的系统边界"><ArchitectureFigure/></DataComponent>}
    <Narrative name="concepts"/>
    <Narrative name="deployment" queries={["deployment"]}/>
    <Narrative name="api" queries={["contracts"]}/>
    {table("table-contracts","contracts","表1 · 当前云端推理API",[{field:"operation",label:"用途"},{field:"endpoint",label:"方法与路径"},{field:"payload",label:"输入契约"},{field:"result",label:"输出/含义"}],"云端独立推理API；与旧版业务API区分")}
    <Narrative name="client" queries={["contracts"]}/>
    <Narrative name="data" queries={["inventory"]}/>
    {table("table-inventory","inventory","表2 · 原始素材、派生产物与向量口径",[{field:"kind",label:"对象"},{field:"count",label:"数量"},{field:"unit",label:"单位"},{field:"meaning",label:"含义"}],"素材与向量计数；不同对象不可相加成样本量")}
    <FlowExplorer/>
    <Narrative name="encoding" queries={["inventory","schema"]}/>
    {table("table-schema","schema","表3 · SQLite数据模型",[{field:"table",label:"表"},{field:"role",label:"职责"},{field:"fields",label:"关键字段"}],"八张表的存储职责")}
    <Narrative name="jobs" queries={["jobs"]}/>
    {table("table-jobs","jobs","表4 · 既有任务耗时记录",[{field:"task",label:"任务"},{field:"elapsed_s",label:"墙钟时间 / s",renderCell:v=>number(v,3)},{field:"scope",label:"适用口径"}],"导入、首次编码与复用续跑；非同协议吞吐对照")}
    <Narrative name="retrieval" queries={["routes"]}/>
    {table("table-routes","routes","表5 · 5种文字召回模式",[{field:"mode",label:"模式"},{field:"input",label:"候选表示"},{field:"candidate_count",label:"当前候选量"},{field:"gpu",label:"GPU调用"}],"各路候选与GPU角色；评分算法见数据来源")}
    <Narrative name="case" queries={["case","similarity"]}/>
    {visible("figure-case")&&<DataComponent id="figure-case" kind="custom" queryId="case" queryIds={["case","frames"]} sourceRowsByQuery={{case:rows("case"),frames:rows("frames")}} displayRows={rows("case")} title="图3 · 真实封面、查询与8张实际采样帧"><CaseFigure caseRow={rows("case")[0]} frames={rows("frames")}/></DataComponent>}
    <Narrative name="caseTiming" queries={["trace_timing"]}/>
    {visible("chart-trace")&&<EvidenceChart id="chart-trace" queryId="trace_timing" title="图4 · 单次检索的耗时主要落在结果读取" rows={rows("trace_timing")} sourceRows={rows("trace_timing")} spec={timingSpec} height={260}/>}
    <Narrative name="time" queries={["frames","case","galaxy"]}/>
    <Narrative name="whitebox" queries={["schema","routes"]}/>
    <Narrative name="quality" queries={["quality"]}/>
    {table("table-quality","quality","表6 · 7个质量任务的768维主结果",[{field:"task",label:"任务"},{field:"queries",label:"查询数"},{field:"corpus",label:"候选数"},{field:"metric_value",label:"nDCG@10（0–1）",renderCell:v=>number(v,4)},{field:"caveat",label:"协议与限制"}],"每行按其候选与标注口径解释；不能跨任务直接排名")}
    <Narrative name="hybrid" queries={["hybrid"]}/>
    {table("table-hybrid","hybrid","表7 · 固定两路RRF的负结果",[{field:"task",label:"任务"},{field:"dense768",label:"纯向量 nDCG@10",renderCell:v=>number(v,4)},{field:"char_tfidf_rrf",label:"向量+TF-IDF RRF",renderCell:v=>number(v,4)},{field:"delta_pp",label:"变化 / 百分点",renderCell:v=>number(v,2)}],"固定两路探索实验；不同于Demo的四路RRF")}
    <Narrative name="validation" queries={["validation"]}/>
    <Narrative name="performance" queries={["capacity","performance_guidance"]}/>
    {visible("chart-capacity")&&<EvidenceChart id="chart-capacity" queryId="capacity" title="图5 · 增加请求并发没有提高有效吞吐" spec={throughputSpec} rows={rows("capacity")} sourceRows={rows("capacity")} height={225}/>}
    {table("table-capacity","capacity","表8 · 同一短query负载的成功与限流",[{field:"concurrency",label:"客户端并发"},{field:"successes",label:"成功"},{field:"attempts",label:"全部尝试"},{field:"success_rps",label:"成功请求 / s",renderCell:v=>number(v,2)},{field:"rejected_429_pct",label:"429比例 / %",renderCell:v=>number(v,2)}],"约15秒单轮短测；并发2的429为953/1221")}
    <Narrative name="stability" queries={["soak"]}/>
    {table("table-soak","soak","表9 · 30分钟混合负载的HTTP延迟",[{field:"workload",label:"工作负载"},{field:"requests",label:"请求数"},{field:"successes",label:"成功"},{field:"p50_ms",label:"p50 / ms",renderCell:v=>number(v,2)},{field:"p95_ms",label:"p95 / ms",renderCell:v=>number(v,2)},{field:"p99_ms",label:"p99 / ms",renderCell:v=>number(v,2)}],"固定五类近等量、单在途、重复媒体；单轮经验分位数")}
    <Narrative name="operations" queries={["deployment"]}/>
    <Narrative name="next"/>
    <Narrative name="appendix"/>
    <RichNarrative id="system:footer" className="system-footer" value="本地系统工程总结。证据分为实现核查、历史实测与后续建议；未发表、未声明同行评审或生产级SLA。" label="编辑页尾说明"/>
  </article>;
}
