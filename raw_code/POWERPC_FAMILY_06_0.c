POWERPC_FAMILY(POWER5P)(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    PowerPCCPUClass *pcc = POWERPC_CPU_CLASS(oc);



    dc->fw_name = "PowerPC,POWER5";

    dc->desc = "POWER5+";

    pcc->init_proc = init_proc_power5plus;

    pcc->check_pow = check_pow_970FX;

    pcc->insns_flags = PPC_INSNS_BASE | PPC_STRING | PPC_MFTB |

                       PPC_FLOAT | PPC_FLOAT_FSEL | PPC_FLOAT_FRES |

                       PPC_FLOAT_FSQRT | PPC_FLOAT_FRSQRTE |

                       PPC_FLOAT_STFIWX |

                       PPC_CACHE | PPC_CACHE_ICBI | PPC_CACHE_DCBZ |

                       PPC_MEM_SYNC | PPC_MEM_EIEIO |

                       PPC_MEM_TLBIE | PPC_MEM_TLBSYNC |

                       PPC_64B |

                       PPC_SEGMENT_64B | PPC_SLBI;

    pcc->msr_mask = (1ull << MSR_SF) |

                    (1ull << MSR_VR) |

                    (1ull << MSR_POW) |

                    (1ull << MSR_EE) |

                    (1ull << MSR_PR) |

                    (1ull << MSR_FP) |

                    (1ull << MSR_ME) |

                    (1ull << MSR_FE0) |

                    (1ull << MSR_SE) |

                    (1ull << MSR_DE) |

                    (1ull << MSR_FE1) |

                    (1ull << MSR_IR) |

                    (1ull << MSR_DR) |

                    (1ull << MSR_PMM) |

                    (1ull << MSR_RI);

    pcc->mmu_model = POWERPC_MMU_64B;

#if defined(CONFIG_SOFTMMU)

    pcc->handle_mmu_fault = ppc_hash64_handle_mmu_fault;

#endif

    pcc->excp_model = POWERPC_EXCP_970;

    pcc->bus_model = PPC_FLAGS_INPUT_970;

    pcc->bfd_mach = bfd_mach_ppc64;

    pcc->flags = POWERPC_FLAG_VRE | POWERPC_FLAG_SE |

                 POWERPC_FLAG_BE | POWERPC_FLAG_PMM |

                 POWERPC_FLAG_BUS_CLK;

    pcc->l1_dcache_size = 0x8000;

    pcc->l1_icache_size = 0x10000;

}
