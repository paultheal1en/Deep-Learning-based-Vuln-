i2c_slave *twl92230_init(i2c_bus *bus, qemu_irq irq)

{

    struct menelaus_s *s = (struct menelaus_s *)

            i2c_slave_init(bus, 0, sizeof(struct menelaus_s));



    s->i2c.event = menelaus_event;

    s->i2c.recv = menelaus_rx;

    s->i2c.send = menelaus_tx;



    s->irq = irq;

    s->rtc.hz = qemu_new_timer(rt_clock, menelaus_rtc_hz, s);

    s->in = qemu_allocate_irqs(menelaus_gpio_set, s, 3);

    s->pwrbtn = qemu_allocate_irqs(menelaus_pwrbtn_set, s, 1)[0];



    menelaus_reset(&s->i2c);



    register_savevm("menelaus", -1, 0, menelaus_save, menelaus_load, s);



    return &s->i2c;

}
