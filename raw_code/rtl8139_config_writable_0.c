static int rtl8139_config_writable(RTL8139State *s)

{

    if (s->Cfg9346 & Cfg9346_Unlock)

    {

        return 1;

    }



    DPRINTF("Configuration registers are write-protected\n");



    return 0;

}
