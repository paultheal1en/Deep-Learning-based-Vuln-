static void arm_thistimer_write(void *opaque, target_phys_addr_t addr,

                                uint64_t value, unsigned size)

{

    arm_mptimer_state *s = (arm_mptimer_state *)opaque;

    int id = get_current_cpu(s);

    timerblock_write(&s->timerblock[id * 2], addr, value, size);

}
