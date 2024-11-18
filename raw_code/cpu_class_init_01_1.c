static void cpu_class_init(ObjectClass *oc, void *data)

{

    SCLPEventClass *k = SCLP_EVENT_CLASS(oc);

    DeviceClass *dc = DEVICE_CLASS(oc);



    k->get_send_mask = send_mask;

    k->get_receive_mask = receive_mask;

    k->read_event_data = read_event_data;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);







}