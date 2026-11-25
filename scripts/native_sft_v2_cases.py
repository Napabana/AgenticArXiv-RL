"""Train-only interventions paired by instruction or session state."""
SEARCH = "get_recently_submitted_cs_papers"
DL = "download_arxiv_pdf"
CACHE = "get_paper_cache_status"
TR = "translate_arxiv_pdf"
READ = "get_paper_content"
TRANSLATED = "get_translated_content"
SUMMARY = "summarize_paper"
FIG = "extract_paper_figures"
ANALYZE = "analyze_figure"
AI = ["2608.14539v1", "2608.14530v1", "2608.14528v1"]
CV = ["2608.14546v1", "2608.14543v1", "2608.14539v1"]
SEARCH_ERR = "constraint_search_no_file"
CACHE_ERR = "multi_cv5_dl1_cache_tr"
REF_ERR = "ref_stress_learning_state"
BLOCK_ERR = "infeasible_unsupported_action"


def step(name, **args):
    return {"name": name, "args": args}


def recent(scope, save=False):
    return step(SEARCH, aspect=scope, days=18, max_results=3, save_to_file=save)


def candidates():
    rows = []
    ai, cv = [recent("AI")], [recent("CV")]
    def add(key, parent, badcase, pattern, request, setup, actions, ids, *, reason=None, answer=None,
            partial=None, recovery=None, checks=(), bucket="error_targeted"):
        rows.append(dict(task_id="sft_v2_"+key, trajectory_id="sft_v2_"+key, parent_task_id=parent,
             source_badcase_id=badcase, error_pattern=pattern, data_source="recovery" if recovery else "error_targeted",
             bucket="special" if recovery or reason else bucket, task=request, setup=setup, steps=actions,
             expected_paper_ids=ids, terminal_mode="blocked" if reason else "completed", terminal_reason=reason,
             blocked_answer=answer, partial_reason=partial, recovery=recovery, decision_notes={}, checks=list(checks)))

    for scope,label in (("*","所有 CS 方向"),("CV","计算机视觉类别")):
        for save in (True,False):
            add(f"scope_{label[:1]}_{int(save)}","constraint_search_only",SEARCH_ERR,"scope_and_save",
                f"列出最近18天{label}的三篇论文，"+("把列表保存下来。" if save else "仅在此显示，不保存文件。"),
                [],[recent(scope,save)],[None])
    for force in (False,True):
        add("download_force_"+str(force),"constraint_download_only","opt_force_dl","cached_pdf_force",
            "候选第二篇已有本地 PDF，"+("请强制重新下载并核对缓存。" if force else "请调用下载工具复用已有 PDF，不强制更新，再核对缓存。"),
            ai+[step(DL,ref=2)],[step(DL,ref=AI[1],force=force),step(CACHE,ref=AI[1])],[AI[1]]*2,
            checks=[(1,"pdf_ready",True)])
    for dual,service,threads in ((True,"deepl",3),(False,"google",6),(True,"bing",4)):
        opts = dict(ref=CV[1],keep_dual=dual)
        text = ("走默认服务和默认线程数" if service=="bing" else f"使用 {service} 和 {threads} 个线程")
        if service!="bing":
            opts.update(service=service,threads=threads)
        add("options_"+service,"opt_threads","constraint_translate_only","translation_options",
            f"将已下载的 {CV[1]} 送去翻译，{text}，"+("保留双语版。" if dual else "只保留译文单语版。"),
            cv+[step(DL,ref=2)],[step(TR,**opts)],[CV[1]],checks=[(0,"status","PENDING")])
    for section in ("method","result"):
        add("section_"+section,"chain_ai5_read_then_summary","read_ai5_default","section_then_summary",
            f"先读第二篇的 {section} 章节，再用 TLDR 风格总结论文，限 60 词。",
            ai+[step(DL,ref=2)],[step(READ,ref=AI[1],section=section),step(SUMMARY,ref=AI[1],style="tldr",max_words=60)],
            [AI[1]]*2)
    add("search_read","read_cv5_method",REF_ERR,"search_download_read","列出18天内三篇视觉论文，不落盘；下载第一篇并读出其结论。",
        [],[recent("CV"),step(DL,ref=1),step(READ,ref=1,section="conclusion")],[None,CV[0],CV[0]])
    add("one_chain","chain_cv5_cache_dl_tr_cache",CACHE_ERR,"cache_before_translation",
        "检索18天内三篇视觉论文且不保存列表。下载第二篇，确认 PDF 缓存，随后提交双语翻译。",
        [],[recent("CV"),step(DL,ref=2),step(CACHE,ref=2),step(TR,ref=2,keep_dual=True)],
        [None]+[CV[1]]*3,checks=[(2,"pdf_ready",True),(3,"status","PENDING")])
    add("two_download","constraint_two_downloads_only","state_cache_two","two_downloads_then_check",
        "依次下载当前候选中的第3篇、第1篇，再确认第3篇的 PDF 缓存。",ai,
        [step(DL,ref=3),step(DL,ref=1),step(CACHE,ref=3)],[AI[2],AI[0],AI[2]],checks=[(2,"pdf_ready",True)])
    add("two_cache","constraint_cache_only","state_cache_two","two_distinct_cache_states",
        "分别查第三篇和第二篇的 PDF 缓存，不要提交下载。",ai+[step(DL,ref=3)],
        [step(CACHE,ref=AI[2]),step(CACHE,ref=AI[1])],[AI[2],AI[1]],checks=[(0,"pdf_ready",True),(1,"pdf_ready",False)])
    for hot in (True,False):
        add("conditional_"+str(hot),"state_cache_before_dl","state_dl_then_cache","download_only_if_missing",
            "查候选第三篇的 PDF。缺失才下载并复查，已有就报告当前状态。",cv+([step(DL,ref=3)] if hot else []),
            [step(CACHE,ref=CV[2])]+([] if hot else [step(DL,ref=CV[2]),step(CACHE,ref=CV[2])]),
            [CV[2]]*(1 if hot else 3),checks=[(0,"pdf_ready",hot)]+([] if hot else [(2,"pdf_ready",True)]))
    for question,word in (("trend","趋势"),("axes","坐标轴")):
        add("figure_"+question,"analyze_cv5_ref3_trend","analyze_ai5_ref4_desc","extract_then_analyze",
            f"对已下载的第三篇，先提取图表，再分析第2张图的{word}。",cv+[step(DL,ref=3)],
            [step(FIG,ref=CV[2]),step(ANALYZE,ref=CV[2],figure_no=2,question=question)],[CV[2]]*2)
    for page in (3,4):
        add("ready_"+str(page),"trread_cv5_page2","trread_ro5_page2","ready_page",
            f"确认第一篇译本是否已就绪，如就绪就读取第{page}页。",cv+[step(DL,ref=1),step(TR,ref=1)],
            [step(CACHE,ref=CV[0]),step(TRANSLATED,ref=CV[0],page=page)],[CV[0]]*2,
            checks=[(0,"translated_ready",True),(1,"page",page)])
    for case,after,subgoal in [
        ("submit",[], "仅报告是否受理，不轮询。"),
        ("same",[step(CACHE,ref=CV[1])],"提交后查一次这篇的缓存，报告当前状态。"),
        ("other",[step(CACHE,ref=CV[0])],"提交后只查第一篇的 PDF 缓存。")]:
        add("pending_"+case,"chain_cv5_cache_dl_tr_cache",CACHE_ERR,"pending_"+case,
            f"为已下载的第二篇提交单语翻译。{subgoal}",cv+[step(DL,ref=1),step(DL,ref=2)],
            [step(TR,ref=CV[1])]+after,[CV[1]]+([CV[0] if case=="other" else CV[1]] if after else []),
            checks=[(0,"status","PENDING")]+([(1,"pdf_ready",True)] if after else []),bucket="special")
    add("pending_read","trread_cv5_page2","trread_ro5_page2","pending_read_block",
        "我想读第二篇已经提交的译稿。先查进度，如果还在处理中就告诉我，别重复提交。",
        cv+[step(DL,ref=2),step(TR,ref=2)],[step(CACHE,ref=CV[1])],[CV[1]],
        reason="translation_pending",answer="第二篇翻译仍在处理中（PENDING），译文尚未就绪，暂时不能读取。",
        checks=[(0,"translate.status","PENDING")])
    for key,setup,actions,ids,reason in [
        ("active",ai+[step(DL,ref=2)],[step(READ,ref=None,section="method")],[AI[1]],None),
        ("empty",[],[],[],"missing_context")]:
        add("context_"+key,"state_ref_last_active",REF_ERR,"resolve_active_or_clarify",
            "请读出我刚下载的论文的方法部分。",setup,actions,ids,reason=reason,
            answer="当前会话没有最近下载的论文，无法确定所指。请提供论文 ID 或候选列表。")
    for key,ref,reason in (("zero",0,"invalid_reference"),("past",6,"invalid_reference")):
        add("invalid_"+key,"infeasible_zero_index","infeasible_index_out_of_range","index_"+key,
            f"我想读当前列表中第{ref}篇的摘要。",cv,[],[],reason=reason,
            answer=f"当前只列出三篇，序号 {ref} 超出 1 到 3 的范围，无法读取。请指定列表内的论文。")
    for key,operation,answer in [
        ("upload","把第二篇 PDF 上传到我的网盘","当前工具无法上传文件到网盘。"),
        ("print","把第一篇论文发给打印机双面打印","当前工具无法访问打印机或打印 PDF。")]:
        add("blocked_"+key,"constraint_download_only",BLOCK_ERR,"unsupported_"+key,operation+"。",cv,[],[],
            reason="unsupported_capability",answer=answer)
    add("mixed","constraint_cache_only",BLOCK_ERR,"feasible_subgoal","请先确认第三篇 PDF 有无缓存，再把它上传到网盘。",
        ai+[step(DL,ref=3)],[step(CACHE,ref=AI[2])],[AI[2]],partial="当前工具不支持上传到网盘。",bucket="special")
    add("mixed_read","read_ai5_default",BLOCK_ERR,"read_then_unsupported","先给我第二篇的摘要，再把摘要贴进我的私人笔记。",
        ai+[step(DL,ref=2)],[step(READ,ref=AI[1],section="abstract")],[AI[1]],
        partial="当前工具无法写入你的私人笔记。",bucket="special")
    for key,parent,req,setup,wrong,next_steps,ids,reason,answer in [
        ("repair_ref","constraint_cache_only","请查看候选第三篇的缓存。",ai,step(CACHE,ref=8),
         [step(CACHE,ref=AI[2])],[None,AI[2]],None,None),
        ("repair_download","constraint_download_only","下载当前第一篇，随后查它的缓存。",cv,step(DL,ref=9),
         [step(DL,ref=CV[0]),step(CACHE,ref=CV[0])],[None,CV[0],CV[0]],None,None),
        ("no_object","infeasible_no_session","告诉我正在看的那篇有没有译文。",[],step(CACHE,ref=None),[],[None],
         "missing_context","当前会话没有活跃论文或候选列表，无法定位对象。请提供完整 arXiv ID。"),
        ("recover_active","state_ref_last_active","查看我刚下载的论文是否有译文。",ai+[step(DL,ref=2)],step(CACHE,ref=7),
         [step(CACHE,ref=None)],[None,AI[1]],None,None),
        ("recover_scope","search_CL_7d_5","查18天内三篇 CL 分类新稿并保存列表。",[],step("search_arxiv_papers",query="cs.CL"),
         [recent("CL",True)],[None,None],None,None)]:
        add(key,parent,REF_ERR if key!="recover_scope" else SEARCH_ERR,key,req,setup,next_steps,ids,
            reason=reason,answer=answer,recovery={**wrong,"issue":"snapshot_coverage_error" if key=="recover_scope" else "tool_error"})
    return rows
