static void do_cpu_reset(void *opaque)

{

    ARMCPU *cpu = opaque;

    CPUARMState *env = &cpu->env;

    const struct arm_boot_info *info = env->boot_info;



    cpu_reset(CPU(cpu));

    if (info) {

        if (!info->is_linux) {

            /* Jump to the entry point.  */

            env->regs[15] = info->entry & 0xfffffffe;

            env->thumb = info->entry & 1;

        } else {

            if (CPU(cpu) == first_cpu) {

                env->regs[15] = info->loader_start;

                if (!info->dtb_filename) {

                    if (old_param) {

                        set_kernel_args_old(info);

                    } else {

                        set_kernel_args(info);

                    }

                }

            } else {

                info->secondary_cpu_reset_hook(cpu, info);

            }

        }

    }

}
