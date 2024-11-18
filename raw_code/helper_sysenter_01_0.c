void helper_sysenter(CPUX86State *env)

{

    if (env->sysenter_cs == 0) {

        raise_exception_err(env, EXCP0D_GPF, 0);

    }

    env->eflags &= ~(VM_MASK | IF_MASK | RF_MASK);

    cpu_x86_set_cpl(env, 0);



#ifdef TARGET_X86_64

    if (env->hflags & HF_LMA_MASK) {

        cpu_x86_load_seg_cache(env, R_CS, env->sysenter_cs & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK |

                               DESC_L_MASK);

    } else

#endif

    {

        cpu_x86_load_seg_cache(env, R_CS, env->sysenter_cs & 0xfffc,

                               0, 0xffffffff,

                               DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                               DESC_S_MASK |

                               DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK);

    }

    cpu_x86_load_seg_cache(env, R_SS, (env->sysenter_cs + 8) & 0xfffc,

                           0, 0xffffffff,

                           DESC_G_MASK | DESC_B_MASK | DESC_P_MASK |

                           DESC_S_MASK |

                           DESC_W_MASK | DESC_A_MASK);

    env->regs[R_ESP] = env->sysenter_esp;

    env->eip = env->sysenter_eip;

}
