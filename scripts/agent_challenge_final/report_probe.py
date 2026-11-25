"""Render task-level probe evidence after independent reconstruction of every group."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from analyze_probe import analyze
from audit_snapshot import DATA
from audit_tasks import lines
from benchmark.metrics import _parse_tool_action, resolved_paper_id


def rate(k,n):
    return f"{k}/{n} ({k/n:.1%})" if n else "无可判定样本"


def report(path,output):
    out=analyze(path)
    tasks={t["id"]:t for t in lines(DATA/"train_candidates.jsonl")}
    groups=[]
    times=[]
    caps=Counter()
    refs=Counter()
    first_errors=Counter()
    for f in sorted((path/"groups").glob("*/*.json.gz")):
        g=json.loads(gzip.decompress(f.read_bytes()))
        groups.append(g)
        times.append((f.stat().st_mtime-g["seconds"],f.stat().st_mtime))
        for s,r in zip(g["trajectories"],g["credit"]):
            if r["strict_success"] is not None and r["first_error_index"] is not None:
                step=r["first_error_index"]
                kind=r["first_error_kind"]
                first_errors[kind]+=1
                if kind=="wrong_ref":
                    history=s["history"][step]
                    action=_parse_tool_action(history["action"])
                    pos=r["decisions"][step]["verified_prefix_length"]
                    t=tasks[g["task_id"]]
                    refs[(action["name"],str(t["expected_tool_args"][pos].get("ref")),
                          str(action.get("args",action.get("parameters",{})).get("ref")),
                          t["expected_paper_ids"][pos],str(resolved_paper_id(history["observation"])))]+=1
            if r["strict_success"] is None:
                issues=[h for h in s["history"] if h.get("tool_issue") in
                        ("snapshot_coverage_error","environment_execution_error","setup_failure")]
                cause=issues[-1]["observation"] if issues else r["invalid_reason"]
                caps[cause]+=1
    bits=out["totals"]; n=out["groups"]; dec=bits["decisions"]; samples=n*4
    labels=["入选任务 ≥12","入选家族 ≥4","最大家族占比 ≤40%","Strict 混合组 ≥30%","四次以上调用的入选任务 ≥8",
            "含状态分支或跨工具依赖的入选任务 ≥8","DFA 非零覆盖 ≥10%","Anchor 非零覆盖 ≥10%","环境无效组为 0","泄漏审计通过"]
    valid=bits["valid"]; selected=[tasks[tid] for tid in out["selected"]]
    measurements=[str(len(selected)),str(len(out["families"])),rate(max(out["families"].values(),default=0),len(selected)),
                  rate(bits["mixed"],n),str(sum(len(t["expected_tools"])>=4 for t in selected)),
                  str(sum(t["state_branch"] or t["cross_tool_dependency"] for t in selected)),rate(bits["dfa_decisions"],dec),
                  rate(bits["anchor_decisions"],dec),rate(n-valid,n),"0 个完整行为签名冲突，操作目标 ID 跨 split 交集为空"]
    body=["# Challenge Frozen Probe","",f"**{out['decision']}**。冻结 Native SFT v2，在 {len(tasks)} 个训练候选上各采 8 组，每组 G=4，共 {n} 组、{samples} 条轨迹。","",
          "| 准入条件 | 实测 | 通过 |","|---|---|---|"]
    for label,m,ok in zip(labels,measurements,out["conditions"].values(),strict=True):
        body.append(f"| {label} | {m} | {'是' if ok else '否'} |")
    salvaged=sum(s==1 for g in groups for s in g["signals"]["strict"])
    inv=sum(s is None for g in groups for s in g["signals"]["strict"])
    duration=max(b for a,b in times)-min(a for a,b in times)
    tokens=sum(s["assistant_tokens"] for g in groups for s in g["trajectories"])
    peak=max(g["peak_allocated"] for g in groups)
    b=out["totals"]
    body+=["","## 信号与运行","","| 信号 | 全部采样组 | 环境有效组 |","|---|---|---|"]
    for k,title in (("mixed","Strict 混合"),("dfa","同 DFA 状态的正确动作与首错"),("anchor","同真实 Anchor 的不同动作与不同回报")):
        body.append(f"| {title} | {rate(b[k],n)} | {rate(b[k],valid)} |")
    body+=["","## 排除环境无效组后的敏感性分析","",
           f"保留全部原始记录，仅从分析分母排除 {n-valid} 个环境无效组。剩余 {valid} 组中 Strict 混合为 {rate(b['mixed'],valid)}，与原定 30% 门槛对照。DFA 非零决策覆盖为 {rate(b['dfa_decisions'],dec)}，Anchor 非零决策覆盖为 {rate(b['anchor_decisions'],dec)}。这两项覆盖率原本就只在有效组计算。",
           "", "下表保留原定的组内对照次数与截断比例，只放宽整题环境异常排除条件。各题有效组数会不同，表中记录实际曝光量。原设计的 Go 决议单列保留。","",
           "| 题目 | 保留组数 | Strict 混合组 | DFA 对照组 | 截断轨迹 | 按有效组重算可入选 |","|---|---:|---:|---:|---:|---|"]
    for tid in tasks:
        kept=[g["signals"] for g in groups if g["task_id"]==tid and g["signals"]["valid"]]
        mix=sum(g["mixed"] for g in kept)
        dfa=sum(g["dfa"] for g in kept)
        trunc=sum(g["truncated"] for g in kept)
        eligible=len(kept)>0 and max(mix,dfa)>=2 and trunc<=len(kept)
        body.append(f"| {tid} | {len(kept)} | {mix} | {dfa} | {trunc} | {'是' if eligible else '否'} |")
    failures_only=sum(g["signals"]["strict"]==[0,0,0,0] for g in groups)
    dfa_only=sum(g["signals"]["strict"]==[0,0,0,0] and g["signals"]["dfa"] for g in groups)
    ceiling=sum(g["signals"]["strict"]==[1,1,1,1] for g in groups)
    body+=["",f"环境有效组中，{failures_only} 组为四条全部失败，{ceiling} 组为四条全部成功。在全部失败组中仍有 {dfa_only} 组出现同状态的正确前缀与首错。组内终局回报相同，Strict 和依赖回报差异的 Anchor 在这些组没有对照。"]
    body+=["",f"环境可评分轨迹的 Strict 成功为 {rate(salvaged,samples-inv)}。环境无效 {inv} 条；触及轮数或 Token 上限 {b['truncated']} 条。原始 DFA/Anchor 非零决策的分母为环境有效组的 {dec} 个可判定决策。投影 E4 后理论上带非零优势且掩码开启的 Assistant Token 为 {b['nonzero_effective_tokens']}；冻结 Probe 实际参数更新次数为 0。",
           "",f"首组开始到末组落盘约 {duration/3600:.2f} 小时，合计生成 {tokens} 个 Assistant Token；吞吐 {tokens/duration:.1f} Token/秒。单采样进程最高 CUDA allocated 为 {peak/2**30:.2f} GiB。原始数据分组保存，每组均核对任务、Seed、SHA 和信用诊断重算结果。",
           "","## 逐题","","| 任务 | 组 | Strict 混合 | DFA 对照 | Anchor 对照 | 环境无效组 | 截断轨迹 | Strict 成功 | 入选 |","|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tid,d in out["tasks"].items():
        body.append(f"| {tid} | 8 | {d['mixed']} | {d['dfa']} | {d['anchor']} | {8-d['valid']} | {d['truncated']} | {d['strict_success']} | {'是' if d['eligible'] else '否'} |")
    body+=["","## 每题追加采样实际观察到什么","","使用相同的候选题和预定 Seed，仅截取各题前 k 个组。环境无效轨迹从严格成功分母中剔除。","",
           "| 每题采样 | 至少一次严格成功的题 | 成功轨迹 / 可评分轨迹 | 出现 DFA 对照的题 | 出现 Anchor 对照的题 |",
           "|---|---:|---:|---:|---:|"]
    for k in (1,2,4,8):
        sample=[g for g in groups if g["group"]<k]
        outcomes=[x for g in sample for x in g["signals"]["strict"] if x is not None]
        success={g["task_id"] for g in sample if 1 in g["signals"]["strict"]}
        dfa={g["task_id"] for g in sample if g["signals"]["dfa"]}
        anchor={g["task_id"] for g in sample if g["signals"]["anchor"]}
        body.append(f"| {k} 组 / {4*k} 条 | {len(success)} | {rate(sum(outcomes),len(outcomes))} | {len(dfa)} | {len(anchor)} |")
    body+=["","## 首错与引用","", "只计可评分轨迹中的第一个可验证错误，同一条轨迹至多计一次。",""]
    for kind,count in first_errors.most_common():
        body.append(f"- {kind}：{count} 条。")
    if refs:
        body+=["","| 工具 | 指定序号 | 实际引用 | 应解析到论文 | 实际解析到论文 | 次数 |","|---|---:|---|---|---|---:|"]
        for (tool,expected,actual,wanted,observed),count in refs.most_common(12):
            body.append(f"| {tool} | {expected} | {actual} | {wanted} | {observed} | {count} |")
    if caps:
        body+=["","## 环境异常","","次数按轨迹统计。同组另外三条轨迹保留原始判定，含环境异常的组不计入三类有效信号。",""]
        for cause,count in caps.most_common(8):
            body.append(f"- {count} 条。{str(cause).partition('。请先运行')[0][:250]}")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(("\n".join(body)+"\n").encode())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    report(a.directory,a.output)
