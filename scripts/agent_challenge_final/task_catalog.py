"""Ordered, user-requested workflows. Paper roles are resolved from the actual recording."""
import json
from benchmark.task_spec import TaskSpec, Step
from audit_snapshot import DATA, RECENT

DL="download_arxiv_pdf"
CACHE="get_paper_cache_status"
READ="get_paper_content"
SUM="summarize_paper"
FIG="extract_paper_figures"
ANA="analyze_figure"
TR="translate_arxiv_pdf"
PAGE="get_translated_content"

# Recorded pools and one-based ranks. All action targets stay disjoint across splits.
PROFILES={"train":("AI",2,4,"CL",1), "dev":("CV",1,2,"RO",1), "sealed_test":("LG",1,4,"CL",2)}
PARENTS={"A":"chain_ai5_read_then_summary","B":"constraint_cache_only","C":"chain_ro5_dl_three",
         "D":"analyze_cv5_ref1_desc","E":"constraint_search_only","F":"chain_cv5_cache_dl_tr_cache"}

# Each instruction states the order whenever the scorer requires it. Q and X establish
# or replace the list; other tokens address the original paper identities A, B and Z.
RECIPES = {
"train":[
("A","Q","dA cA rA sA","依次下载甲、核实文件缓存、读摘要段，再给甲生成 120 字内的结构化摘要。"),
("A","Q","dB fB gB cB","先下载乙，取出图表并分析图1的趋势，最后核实乙的缓存。"),
("A","","Q dA cA dB fB","查{main}近30天前{n}篇，仅展示列表。依次下载甲、核实甲的缓存，再下载乙并提取乙的图表。"),
("A","Q","dA rA dB rB sB","先下载甲并读取摘要段，然后下载乙、读取乙的摘要段，最后给乙作结构化短摘要（120字）。"),
("A","Q","dB cB tB cA","下载乙，核对乙的缓存并提交单语翻译，随后独立查甲的缓存，再分别告知状态。"),
("A","Q dA","rA sA dB fB gB","甲已备好，读甲的摘要段并生成120字内结构化摘要；随后下载乙，提取图表并解读乙的图1趋势。"),
("B","Q","cA dA cB dB rB","依序检查甲和乙的PDF缓存。每查完一篇，只在缺失时补下载，然后才处理下一篇。最后读取乙的摘要段。"),
("B","Q dA","cA cB dB rB","依序检查甲和乙的PDF缓存。每查完一篇，只在缺失时补下载，然后才处理下一篇。最后读取乙的摘要段。"),
("B","Q dB","cA dA cB rB","依序检查甲和乙的PDF缓存。每查完一篇，只在缺失时补下载，然后才处理下一篇。最后读取乙的摘要段。"),
("B","Q dA dB","cA cB rB","依序检查甲和乙的PDF缓存。每查完一篇，只在缺失时补下载，然后才处理下一篇。最后读取乙的摘要段。"),
("B","Q dB","cB DB cB sB","查看乙的缓存，明确要求强制重新下载，即使已存在也要替换；重新核实缓存后产出120字结构化摘要。"),
("B","Q dA","cA rA cB","先查看甲的缓存，有文件则读甲的摘要段；最后只查乙的缓存并报告，别下载乙。"),
("C","Q","dA X dZ Q cA","下载甲，然后把列表切到{alt}近30天前{z}篇，仅显示。下载新列表末篇，再把显示列表恢复到原检索，最后检查原先甲的缓存。"),
("C","Q dB","rB X cZ Q sB","读乙的摘要段，再切到{alt}近30天前{z}篇，仅显示。查新列表末篇缓存，再恢复原检索列表，给原来的乙生成120字结构化摘要。"),
("C","Q","dB dA X cZ Q cB cA","先下载乙再下载甲，然后显示{alt}近30天前{z}篇并查末篇缓存。恢复原检索列表后，按原乙、原甲的顺序核实两份缓存。"),
("C","Q","dA X dZ fZ Q rA","下载甲，显示{alt}近30天前{z}篇，再下载新列表末篇、取其图表，最后恢复原检索列表并读原甲的摘要段。"),
("D","Q dA dB","rA fB gB sA","先读取甲的摘要段，再提取乙的图表、解读乙的图1趋势，最后为甲生成120字内结构化摘要。"),
("D","Q","dB fB hB rB","下载乙后先提取图表，说明图1的坐标轴，然后读同篇摘要段。"),
("D","Q dB","fB gB sB","取乙的图表，说明图1的趋势，随后产出同篇120字内结构化摘要。"),
("E","","Q dA dB TB cA","显示{main}近30天前{n}篇，不保存列表。按甲、乙顺序下载，为乙提交保留双语的翻译，最后仅核实甲的缓存。"),
("E","Q dA","DA cA dB TB","强制重新下载甲并核实缓存；再正常下载乙，给乙提交双语翻译，到此结束。"),
("E","","QS dB cB rB","检索{main}近30天前{n}篇并保存列表，再下载乙、核实乙的缓存并读取其摘要段。"),
("F","Q dA dB","tA cB rB","提交甲的单语翻译；任务受理后，独立核实乙的PDF缓存、读取乙的摘要段，汇报两篇状态。"),
("F","Q dB tB","cB pB sB","先查乙的缓存；译文就绪时读取第2页，随后为乙生成120字结构化摘要。"),
],
"dev":[
("A","Q","dB rB dA cA sA","依序下载乙、读乙的摘要段、下载甲、核实甲的缓存，最后为甲生成120字结构化摘要。"),
("A","Q dA","fA hA dB cB","先取甲的图表、解释其图1坐标轴，之后下载乙并核实乙的缓存。"),
("B","Q","cB dB fB cA","先核实乙的PDF缓存，缺文件才补下载，再提取乙的图表；最后仅查甲的缓存。"),
("B","Q dB","cB fB cA","先核实乙的PDF缓存，缺文件才补下载，再提取乙的图表；最后仅查甲的缓存。"),
("C","Q dA","X dZ rZ Q cA","显示{alt}近30天前{z}篇。下载新列表末篇并读取其摘要段，随后恢复原检索列表，再查换表前甲的缓存。"),
("D","Q dA","fA gA dB sB","先提取甲的图并解读图1趋势；然后下载乙，为乙生成120字结构化摘要。"),
("E","","QS DA cA dB rB","保存{main}近30天前{n}篇列表，强制下载甲并核实；正常下载乙，再读取乙的摘要段。"),
("F","Q dA","tA dB cB","给甲提交单语翻译。受理后独立下载乙并查乙的缓存，汇报甲翻译的实际状态和乙的PDF状态。"),
],
"sealed_test":[
("A","Q","dA rA fA gA cB","依序下载甲、读取摘要段、取图并分析甲的图1趋势；最后只查乙的PDF缓存。"),
("A","Q","dB sB dA cA fA","先下载乙并生成120字结构化摘要；接着下载甲、核实甲的缓存并提取甲的图表。"),
("B","Q dA","cA rA cB dB sB","先查甲的缓存，已有文件时读摘要段；随后查乙的缓存，缺失才下载；最后为乙作120字结构化摘要。"),
("B","Q dA dB","cA rA cB sB","先查甲的缓存，已有文件时读摘要段；随后查乙的缓存，缺失才下载；最后为乙作120字结构化摘要。"),
("C","Q","dA X dZ cZ Q fA","下载甲后显示{alt}近30天前{z}篇，再下载新列表末篇并查其缓存，恢复原检索列表后提取原甲的图。"),
("C","Q dB","X dZ sZ Q fB hB","先显示{alt}近30天前{z}篇，下载末篇并生成120字结构化摘要，再恢复原检索列表，取原乙的图并说明图1的坐标轴。"),
("C","Q","dB X dZ fZ gZ Q cB","下载乙后显示{alt}近30天前{z}篇。下载新末篇、取图并分析图1趋势，最后恢复原检索列表，核实原乙缓存。"),
("D","Q dA dB","rB fA hA sB cA","读乙的摘要段，再取甲的图并说明图1坐标轴，给乙生成120字结构化摘要，最后核实甲的缓存。"),
("D","Q","dA sA dB fB hB","先下载甲并生成120字结构化摘要；然后下载乙，提取乙的图并说明图1坐标轴。"),
("E","","Q DB cB TB dA rA","只显示{main}近30天前{n}篇，不保存。强制下载乙、核实缓存并提交双语翻译；随后正常下载甲、读取甲的摘要段。"),
("E","Q dB","cB DB rB dA cA","先查乙的缓存、强制重新下载乙并读取摘要段；再正常下载甲并核实甲的缓存。"),
("F","Q dA tA dB","cA pA fB hB","先核实甲的译文缓存；就绪时读取第2页，再独立提取乙的图表、说明乙的图1坐标轴。"),
]}


def build(profiles=PROFILES, catalog=RECIPES):
    inv=json.loads((DATA/"snapshot_inventory.json").read_bytes())
    pools={row["args"]["aspect"]:row["ordered_ids"] for row in inv["search_pools"][RECENT]}
    result=[]
    for split, recipes in catalog.items():
        main,i,j,alternate,k=profiles[split]
        ids={"A":pools[main][i-1],"B":pools[main][j-1],"Z":pools[alternate][k-1]}
        limit=max(i,j)
        for number,(family,setup,steps,text) in enumerate(recipes,1):
            def expand(token, *, setup_step=False):
                if token in ("Q","QS","X"):
                    scope,cap=(alternate,k) if token=="X" else (main,limit)
                    return Step(RECENT,dict(aspect=scope,days=30,max_results=cap,save_to_file=token=="QS")),None
                tool,role=token[:-1],token[-1]
                schema={"d":(DL,{}),"D":(DL,{"force":True}),"c":(CACHE,{}),"r":(READ,{"section":"abstract"}),
                        "s":(SUM,{"style":"structured","max_words":120}),"f":(FIG,{}),
                        "g":(ANA,{"figure_no":1,"question":"trend"}),"h":(ANA,{"figure_no":1,"question":"axes"}),
                        "t":(TR,{"keep_dual":False}),"T":(TR,{"keep_dual":True}),"p":(PAGE,{"page":2})}
                name,args=schema[tool]
                ref = ids[role] if setup_step else str({"A":i, "B":j, "Z":k}[role])
                return Step(name,dict(ref=ref,**args)),ids[role]
            actions=[expand(t) for t in steps.split()]
            initial=[expand(t,setup_step=True)[0] for t in setup.split()]
            aliases=f"甲、乙分别指{('本次检索结果' if not setup else '开始时的候选列表')}的第{i}篇和第{j}篇。"
            wording=aliases+text.format(main=main,alt="全部 CS 分类" if alternate=="*" else alternate,n=limit,z=k)
            tid=f"challenge_{split}_{number:02}_{family}"
            spec=TaskSpec(id=tid,task=wording,setup=tuple(initial),steps=tuple(x for x,_ in actions),
                          category=family,difficulty="L1" if len(actions)<=3 else "L2" if len(actions)==4 else "L3",
                          max_iterations=len(actions)+2)
            row=spec.to_task()
            row.update(family=family,split=split,parent_task_id=PARENTS[family],expected_paper_ids=[p for _,p in actions],
                       state_branch=family=="B",cross_tool_dependency=len({act.tool for act,_ in actions})>=2,
                       subgoals=list(dict.fromkeys((act.tool,p) for act,p in actions if p)),
                       recipe=steps,setup_recipe=setup)
            result.append(row)
    return result
