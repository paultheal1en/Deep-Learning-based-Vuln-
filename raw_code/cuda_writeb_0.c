static void cuda_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    CUDAState *s = opaque;



    addr = (addr >> 9) & 0xf;

    CUDA_DPRINTF("write: reg=0x%x val=%02x\n", (int)addr, val);



    switch(addr) {

    case 0:

        s->b = val;

        cuda_update(s);

        break;

    case 1:

        s->a = val;

        break;

    case 2:

        s->dirb = val;

        break;

    case 3:

        s->dira = val;

        break;

    case 4:

        s->timers[0].latch = (s->timers[0].latch & 0xff00) | val;

        cuda_timer_update(s, &s->timers[0], qemu_get_clock_ns(vm_clock));

        break;

    case 5:

        s->timers[0].latch = (s->timers[0].latch & 0xff) | (val << 8);

        s->ifr &= ~T1_INT;

        set_counter(s, &s->timers[0], s->timers[0].latch);

        break;

    case 6:

        s->timers[0].latch = (s->timers[0].latch & 0xff00) | val;

        cuda_timer_update(s, &s->timers[0], qemu_get_clock_ns(vm_clock));

        break;

    case 7:

        s->timers[0].latch = (s->timers[0].latch & 0xff) | (val << 8);

        s->ifr &= ~T1_INT;

        cuda_timer_update(s, &s->timers[0], qemu_get_clock_ns(vm_clock));

        break;

    case 8:

        s->timers[1].latch = val;

        set_counter(s, &s->timers[1], val);

        break;

    case 9:

        set_counter(s, &s->timers[1], (val << 8) | s->timers[1].latch);

        break;

    case 10:

        s->sr = val;

        break;

    case 11:

        s->acr = val;

        cuda_timer_update(s, &s->timers[0], qemu_get_clock_ns(vm_clock));

        cuda_update(s);

        break;

    case 12:

        s->pcr = val;

        break;

    case 13:

        /* reset bits */

        s->ifr &= ~val;

        cuda_update_irq(s);

        break;

    case 14:

        if (val & IER_SET) {

            /* set bits */

            s->ier |= val & 0x7f;

        } else {

            /* reset bits */

            s->ier &= ~val;

        }

        cuda_update_irq(s);

        break;

    default:

    case 15:

        s->anh = val;

        break;

    }

}
