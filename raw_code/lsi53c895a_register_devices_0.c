static void lsi53c895a_register_devices(void)

{

    type_register_static(&lsi_info);

    type_register_static_alias(&lsi_info, "lsi");

}
