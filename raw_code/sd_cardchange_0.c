static void sd_cardchange(void *opaque, bool load)

{

    SDState *sd = opaque;



    qemu_set_irq(sd->inserted_cb, blk_is_inserted(sd->blk));

    if (blk_is_inserted(sd->blk)) {

        sd_reset(DEVICE(sd));

        qemu_set_irq(sd->readonly_cb, sd->wp_switch);

    }

}
