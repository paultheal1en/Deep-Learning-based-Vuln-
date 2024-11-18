static void guest_panicked(void)

{

    QObject *data;



    data = qobject_from_jsonf("{ 'action': %s }", "pause");

    monitor_protocol_event(QEVENT_GUEST_PANICKED, data);

    qobject_decref(data);



    vm_stop(RUN_STATE_GUEST_PANICKED);

}
