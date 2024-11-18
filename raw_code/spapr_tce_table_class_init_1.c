static void spapr_tce_table_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->vmsd = &vmstate_spapr_tce_table;

    dc->init = spapr_tce_table_realize;

    dc->reset = spapr_tce_reset;



    QLIST_INIT(&spapr_tce_tables);



    /* hcall-tce */

    spapr_register_hypercall(H_PUT_TCE, h_put_tce);


}