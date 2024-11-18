qemu_irq isa_reserve_irq(int isairq)

{

    if (isairq < 0 || isairq > 15) {

        hw_error("isa irq %d invalid", isairq);

    }

    if (isabus->assigned & (1 << isairq)) {

        hw_error("isa irq %d already assigned", isairq);

    }

    isabus->assigned |= (1 << isairq);

    return isabus->irqs[isairq];

}
