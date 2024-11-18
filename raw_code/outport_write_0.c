static void outport_write(KBDState *s, uint32_t val)

{

    DPRINTF("kbd: write outport=0x%02x\n", val);

    s->outport = val;

    if (s->a20_out) {

        qemu_set_irq(*s->a20_out, (val >> 1) & 1);

    }

    if (!(val & 1)) {

        qemu_system_reset_request();

    }

}
