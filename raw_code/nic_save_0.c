static void nic_save(QEMUFile * f, void *opaque)

{

    EEPRO100State *s = opaque;

    int i;



    pci_device_save(&s->dev, f);



    qemu_put_8s(f, &s->rxcr);



    qemu_put_8s(f, &s->cmd);

    qemu_put_be32s(f, &s->start);

    qemu_put_be32s(f, &s->stop);

    qemu_put_8s(f, &s->boundary);

    qemu_put_8s(f, &s->tsr);

    qemu_put_8s(f, &s->tpsr);

    qemu_put_be16s(f, &s->tcnt);

    qemu_put_be16s(f, &s->rcnt);

    qemu_put_be32s(f, &s->rsar);

    qemu_put_8s(f, &s->rsr);

    qemu_put_8s(f, &s->isr);

    qemu_put_8s(f, &s->dcfg);

    qemu_put_8s(f, &s->imr);

    qemu_put_buffer(f, s->phys, 6);

    qemu_put_8s(f, &s->curpag);

    qemu_put_buffer(f, s->mult, 8);

    qemu_put_buffer(f, s->mem, sizeof(s->mem));



    /* Save all members of struct between scv_stat and mem. */

    qemu_put_8s(f, &s->scb_stat);

    qemu_put_8s(f, &s->int_stat);

    for (i = 0; i < 3; i++) {

        qemu_put_be32s(f, &s->region[i]);

    }

    qemu_put_buffer(f, s->macaddr, 6);

    for (i = 0; i < 19; i++) {

        qemu_put_be32s(f, &s->statcounter[i]);

    }

    for (i = 0; i < 32; i++) {

        qemu_put_be16s(f, &s->mdimem[i]);

    }

    /* The eeprom should be saved and restored by its own routines. */

    qemu_put_be32s(f, &s->device);

    qemu_put_be32s(f, &s->pointer);

    qemu_put_be32s(f, &s->cu_base);

    qemu_put_be32s(f, &s->cu_offset);

    qemu_put_be32s(f, &s->ru_base);

    qemu_put_be32s(f, &s->ru_offset);

    qemu_put_be32s(f, &s->statsaddr);

    /* Save epro100_stats_t statistics. */

    qemu_put_be32s(f, &s->statistics.tx_good_frames);

    qemu_put_be32s(f, &s->statistics.tx_max_collisions);

    qemu_put_be32s(f, &s->statistics.tx_late_collisions);

    qemu_put_be32s(f, &s->statistics.tx_underruns);

    qemu_put_be32s(f, &s->statistics.tx_lost_crs);

    qemu_put_be32s(f, &s->statistics.tx_deferred);

    qemu_put_be32s(f, &s->statistics.tx_single_collisions);

    qemu_put_be32s(f, &s->statistics.tx_multiple_collisions);

    qemu_put_be32s(f, &s->statistics.tx_total_collisions);

    qemu_put_be32s(f, &s->statistics.rx_good_frames);

    qemu_put_be32s(f, &s->statistics.rx_crc_errors);

    qemu_put_be32s(f, &s->statistics.rx_alignment_errors);

    qemu_put_be32s(f, &s->statistics.rx_resource_errors);

    qemu_put_be32s(f, &s->statistics.rx_overrun_errors);

    qemu_put_be32s(f, &s->statistics.rx_cdt_errors);

    qemu_put_be32s(f, &s->statistics.rx_short_frame_errors);

    qemu_put_be32s(f, &s->statistics.fc_xmt_pause);

    qemu_put_be32s(f, &s->statistics.fc_rcv_pause);

    qemu_put_be32s(f, &s->statistics.fc_rcv_unsupported);

    qemu_put_be16s(f, &s->statistics.xmt_tco_frames);

    qemu_put_be16s(f, &s->statistics.rcv_tco_frames);

    qemu_put_be32s(f, &s->statistics.complete);

#if 0

    qemu_put_be16s(f, &s->status);

#endif



    /* Configuration bytes. */

    qemu_put_buffer(f, s->configuration, sizeof(s->configuration));

}
