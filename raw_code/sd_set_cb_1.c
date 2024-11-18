void sd_set_cb(SDState *sd, qemu_irq readonly, qemu_irq insert)

{

    sd->readonly_cb = readonly;

    sd->inserted_cb = insert;

    qemu_set_irq(readonly, bdrv_is_read_only(sd->bdrv));

    qemu_set_irq(insert, bdrv_is_inserted(sd->bdrv));

}
