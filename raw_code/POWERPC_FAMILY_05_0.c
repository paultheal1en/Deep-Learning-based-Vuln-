POWERPC_FAMILY(e5500)(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    PowerPCCPUClass *pcc = POWERPC_CPU_CLASS(oc);



    dc->desc = "e5500 core";

    pcc->init_proc = init_proc_e5500;

    pcc->check_pow = check_pow_none;

    pcc->insns_flags = PPC_INSNS_BASE | PPC_ISEL |

                       PPC_WRTEE | PPC_RFDI | PPC_RFMCI |

                       PPC_CACHE | PPC_CACHE_LOCK | PPC_CACHE_ICBI |

                       PPC_CACHE_DCBZ | PPC_CACHE_DCBA |

                       PPC_FLOAT | PPC_FLOAT_FRES |

                       PPC_FLOAT_FRSQRTE | PPC_FLOAT_FSEL |

                       PPC_FLOAT_STFIWX | PPC_WAIT |

                       PPC_MEM_TLBSYNC | PPC_TLBIVAX | PPC_MEM_SYNC |

                       PPC_64B | PPC_POPCNTB | PPC_POPCNTWD;

    pcc->insns_flags2 = PPC2_BOOKE206 | PPC2_PRCNTL | PPC2_PERM_ISA206;

    pcc->msr_mask = (1ull << MSR_CM) |

                    (1ull << MSR_GS) |

                    (1ull << MSR_UCLE) |

                    (1ull << MSR_CE) |

                    (1ull << MSR_EE) |

                    (1ull << MSR_PR) |

                    (1ull << MSR_FP) |

                    (1ull << MSR_ME) |

                    (1ull << MSR_FE0) |

                    (1ull << MSR_DE) |

                    (1ull << MSR_FE1) |

                    (1ull << MSR_IR) |

                    (1ull << MSR_DR) |

                    (1ull << MSR_PX) |

                    (1ull << MSR_RI);

    pcc->mmu_model = POWERPC_MMU_BOOKE206;

    pcc->excp_model = POWERPC_EXCP_BOOKE;

    pcc->bus_model = PPC_FLAGS_INPUT_BookE;

    /* FIXME: figure out the correct flag for e5500 */

    pcc->bfd_mach = bfd_mach_ppc_e500;

    pcc->flags = POWERPC_FLAG_CE | POWERPC_FLAG_DE |

                 POWERPC_FLAG_PMM | POWERPC_FLAG_BUS_CLK;

}
