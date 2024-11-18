static void jazz_led_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned int size)

{

    LedState *s = opaque;

    uint8_t new_val = val & 0xff;



    trace_jazz_led_write(addr, new_val);



    s->segments = new_val;

    s->state |= REDRAW_SEGMENTS;

}
