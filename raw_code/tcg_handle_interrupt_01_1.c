static void tcg_handle_interrupt(CPUArchState *env, int mask)

{

    CPUState *cpu = ENV_GET_CPU(env);

    int old_mask;



    old_mask = env->interrupt_request;

    env->interrupt_request |= mask;



    /*

     * If called from iothread context, wake the target cpu in

     * case its halted.

     */

    if (!qemu_cpu_is_self(cpu)) {

        qemu_cpu_kick(cpu);

        return;

    }



    if (use_icount) {

        env->icount_decr.u16.high = 0xffff;

        if (!can_do_io(env)

            && (mask & ~old_mask) != 0) {

            cpu_abort(env, "Raised interrupt while not in I/O function");

        }

    } else {

        cpu_unlink_tb(cpu);

    }

}
