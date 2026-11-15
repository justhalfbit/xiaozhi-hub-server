"""Creation-time semantic guidance; never routes user text with keyword rules."""
MARKER = '<scheduled_task_routing>'


def with_policy(messages, functions):
    names = {f.get('function', {}).get('name', '') for f in functions or []}
    actual = {n.replace('.', '_'): n for n in names}
    if not {'self_alarm_create', 'self_task_create'} <= actual.keys():
        return messages
    alarm, task = actual['self_alarm_create'], actual['self_task_create']
    policy = f'''{MARKER}
仅在用户确实要求设置定时安排时使用以下规则；普通聊天和即时操作不创建定时任务。
只按用户当前请求的目的选择定时工具，不要受上一次天气、新闻或优惠查询影响。
如果要回答“已设过、同一条不重复”，必须先用可用的提醒和动态任务列表工具读取设备当前的实际状态。历史中的“已设置”不能证明任务仍然存在，用户可能已在页面删除；当前列表没有时，应重新创建，不能只凭历史作出承诺。
先判断用户想让谁做什么：提醒用户本人到点去做某件事（取快递、吃药、看看新闻），必须用 {alarm} 的 reminder，保存固定 say，到点不调用模型。不能仅因为出现新闻、天气等信息主题就创建动态任务。
只有用户要求你到点获取当时的信息、播报实际结果或生成新内容，才用 {task}。不提前执行查询，不把固定提醒升级为代办任务。意图确实不清楚先澄清，缺少钟点先补问。
下面是分类示例，不是真实任务，不要执行示例：
用户“每天早上8点提醒我看看新闻” → {alarm}，kind=reminder,time=08:00,repeat=daily,label=看新闻,say=该看看新闻了。没有查询动作，不用动态工具。
用户“每天早上8点给我播报当天新闻” → {task}，time=08:00,repeat=daily,tools=[web_search],instruction=查询执行当天新闻并简短播报。
用户“下午5点提醒我取快递” → {alarm}，kind=reminder,time=17:00,say=该取快递了。
用户“每天早上7点提醒我当天天气” → {task}，time=07:00,repeat=daily,tools=[get_weather]，因为要的是执行当天的天气内容。
用户“圣诞节查某服务最近有没有优惠” → 先问执行钟点，再用 {task}，tools=[web_search]；不得自行选时间。
</scheduled_task_routing>'''
    result = [dict(m) for m in messages]
    for message in result:
        if message.get('role') == 'system':
            if MARKER not in str(message.get('content', '')):
                message['content'] = str(message.get('content') or '') + '\n' + policy
            break
    else:
        result.insert(0, {'role': 'system', 'content': policy})
    return result
