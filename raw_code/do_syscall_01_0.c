static int do_syscall(CPUState *env,

                      struct kqemu_cpu_state *kenv)

{

    int selector;



    selector = (env->star >> 32) & 0xffff;

#ifdef TARGET_X86_64

    if (env->hflags & HF_LMA_MASK) {

        int code64;



        env->regs[R_ECX] = kenv->next_eip;

        env->regs[11] = env->eflags;



        code64 = env->hflags & HF_CS64_MASK;



        cpu_x86_set_cpl(env, 0);

        cpu_x86_load_seg_cache(env, R_CS, selector & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK | DESC_L_MASK);

        cpu_x86_load_seg_cache(env, R_SS, (selector + 8) & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_W_MASK | DESC_A_MASK);

        env->eflags &= ~env->fmask;

        if (code64)

            env->eip = env->lstar;

        else

            env->eip = env->cstar;

    } else

#endif

    {

        env->regs[R_ECX] = (uint32_t)kenv->next_eip;



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

        env->eflags &= ~(IF_MASK | RF_MASK | VM_MASK);

        env->eip = (uint32_t)env->star;

    }

    return 2;

}
