static void quiesce_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SCLPEventClass *k = SCLP_EVENT_CLASS(klass);



    dc->reset = quiesce_reset;

    dc->vmsd = &vmstate_sclpquiesce;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    k->init = quiesce_init;



    k->get_send_mask = send_mask;

    k->get_receive_mask = receive_mask;

    k->can_handle_event = can_handle_event;

    k->read_event_data = read_event_data;

    k->write_event_data = NULL;

}
