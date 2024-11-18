void helper_syscall(CPUX86State *env, int next_eip_addend)

{

    int selector;



    if (!(env->efer & MSR_EFER_SCE)) {

        raise_exception_err(env, EXCP06_ILLOP, 0);

    }

    selector = (env->star >> 32) & 0xffff;

    if (env->hflags & HF_LMA_MASK) {

        int code64;



        env->regs[R_ECX] = env->eip + next_eip_addend;

        env->regs[11] = cpu_compute_eflags(env);



        code64 = env->hflags & HF_CS64_MASK;



        env->eflags &= ~env->fmask;

        cpu_load_eflags(env, env->eflags, 0);

        cpu_x86_set_cpl(env, 0);

        cpu_x86_load_seg_cache(env, R_CS, selector & 0xfffc,

                           0, 0xffffffff,

                               DESC_G_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK |

                               DESC_L_MASK);

        cpu_x86_load_seg_cache(env, R_SS, (selector + 8) & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_W_MASK | DESC_A_MASK);

        if (code64) {

            env->eip = env->lstar;

        } else {

            env->eip = env->cstar;

        }

    } else {

        env->regs[R_ECX] = (uint32_t)(env->eip + next_eip_addend);



        env->eflags &= ~(IF_MASK | RF_MASK | VM_MASK);

        cpu_x86_set_cpl(env, 0);

        cpu_x86_load_seg_cache(env, R_CS, selector & 0xfffc,

                           0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

        cpu_x86_load_seg_cache(env, R_SS, (selector + 8) & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_W_MASK | DESC_A_MASK);

        env->eip = (uint32_t)env->star;

    }

}
