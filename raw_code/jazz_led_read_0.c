static uint64_t jazz_led_read(void *opaque, target_phys_addr_t addr,

                              unsigned int size)

{

    LedState *s = opaque;

    uint8_t val;



    val = s->segments;

    trace_jazz_led_read(addr, val);



    return val;

}
