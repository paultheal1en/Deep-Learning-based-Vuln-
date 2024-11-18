static void css_init(void)

{

    QTAILQ_INIT(&channel_subsys.pending_crws);

    channel_subsys.sei_pending = false;

    channel_subsys.do_crw_mchk = true;

    channel_subsys.crws_lost = false;

    channel_subsys.chnmon_active = false;

    QTAILQ_INIT(&channel_subsys.io_adapters);

    QTAILQ_INIT(&channel_subsys.indicator_addresses);

}
