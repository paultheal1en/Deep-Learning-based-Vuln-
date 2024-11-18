void isa_init_irq(ISADevice *dev, qemu_irq *p, int isairq)

{

    assert(dev->nirqs < ARRAY_SIZE(dev->isairq));

    if (isabus->assigned & (1 << isairq)) {

        hw_error("isa irq %d already assigned", isairq);

    }

    isabus->assigned |= (1 << isairq);

    dev->isairq[dev->nirqs] = isairq;

    *p = isabus->irqs[isairq];

    dev->nirqs++;

}
