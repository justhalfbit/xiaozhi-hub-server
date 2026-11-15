from core.agent.scheduled_policy import with_policy, MARKER


def tools(*names):
    return [{'type':'function','function':{'name':n}} for n in names]


def test_policy_only_for_devices_with_both_capabilities():
    messages=[{'role':'user','content':'提醒我看新闻'}]
    assert with_policy(messages,tools('self_alarm_create')) is messages
    assert with_policy(messages,tools('web_search')) is messages


def test_guidance_is_system_only_and_does_not_pollute_stored_conversation():
    messages=[{'role':'system','content':'原角色'}, {'role':'user','content':'每天提醒我看看新闻'}]
    result=with_policy(messages,tools('self_alarm_create','self_task_create'))
    assert MARKER in result[0]['content']
    assert '本人' in result[0]['content'] and '到点不调用模型' in result[0]['content']
    assert messages[0]['content']=='原角色' and result[1]==messages[1]
    assert len(result)==2


def test_native_names_and_no_duplicate_injection():
    messages=[{'role':'user','content':'天气'}]
    result=with_policy(messages,tools('self.alarm.create','self.task.create'))
    assert result[0]['role']=='system' and 'self.alarm.create' in result[0]['content']
    assert with_policy(result,tools('self.alarm.create','self.task.create'))==result


def test_stale_history_cannot_prove_a_schedule_still_exists():
    result = with_policy([], tools('self_alarm_create', 'self_task_create'))
    assert '读取设备当前的实际状态' in result[0]['content']
    assert '页面删除' in result[0]['content']
