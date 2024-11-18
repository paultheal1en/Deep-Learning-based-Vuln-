int pcnet_common_init(DeviceState *dev, PCNetState *s, NetClientInfo *info)

{

    int i;

    uint16_t checksum;



    s->poll_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, pcnet_poll_timer, s);



    qemu_macaddr_default_if_unset(&s->conf.macaddr);

    s->nic = qemu_new_nic(info, &s->conf, object_get_typename(OBJECT(dev)), dev->id, s);

    qemu_format_nic_info_str(qemu_get_queue(s->nic), s->conf.macaddr.a);



    /* Initialize the PROM */



    /*

      Datasheet: http://pdfdata.datasheetsite.com/web/24528/AM79C970A.pdf

      page 95

    */

    memcpy(s->prom, s->conf.macaddr.a, 6);

    /* Reserved Location: must be 00h */

    s->prom[6] = s->prom[7] = 0x00;

    /* Reserved Location: must be 00h */

    s->prom[8] = 0x00;

    /* Hardware ID: must be 11h if compatibility to AMD drivers is desired */

    s->prom[9] = 0x11;

    /* User programmable space, init with 0 */

    s->prom[10] = s->prom[11] = 0x00;

    /* LSByte of two-byte checksum, which is the sum of bytes 00h-0Bh

       and bytes 0Eh and 0Fh, must therefore be initialized with 0! */

    s->prom[12] = s->prom[13] = 0x00;

    /* Must be ASCII W (57h) if compatibility to AMD

       driver software is desired */

    s->prom[14] = s->prom[15] = 0x57;



    for (i = 0, checksum = 0; i < 16; i++) {

        checksum += s->prom[i];

    }

    *(uint16_t *)&s->prom[12] = cpu_to_le16(checksum);



    s->lnkst = 0x40; /* initial link state: up */



    return 0;

}
