static void wdt_diag288_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    DIAG288Class *diag288 = DIAG288_CLASS(klass);



    dc->realize = wdt_diag288_realize;

    dc->unrealize = wdt_diag288_unrealize;

    dc->reset = wdt_diag288_reset;


    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->vmsd = &vmstate_diag288;

    diag288->handle_timer = wdt_diag288_handle_timer;

}