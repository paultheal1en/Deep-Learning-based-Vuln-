target_ulong do_load_msr (CPUPPCState *env)

{

    return

#if defined (TARGET_PPC64)

        ((target_ulong)msr_sf   << MSR_SF)   |

        ((target_ulong)msr_isf  << MSR_ISF)  |

        ((target_ulong)msr_hv   << MSR_HV)   |

#endif

        ((target_ulong)msr_ucle << MSR_UCLE) |

        ((target_ulong)msr_vr   << MSR_VR)   | /* VR / SPE */

        ((target_ulong)msr_ap   << MSR_AP)   |

        ((target_ulong)msr_sa   << MSR_SA)   |

        ((target_ulong)msr_key  << MSR_KEY)  |

        ((target_ulong)msr_pow  << MSR_POW)  | /* POW / WE */

        ((target_ulong)msr_tgpr << MSR_TGPR) | /* TGPR / CE */

        ((target_ulong)msr_ile  << MSR_ILE)  |

        ((target_ulong)msr_ee   << MSR_EE)   |

        ((target_ulong)msr_pr   << MSR_PR)   |

        ((target_ulong)msr_fp   << MSR_FP)   |

        ((target_ulong)msr_me   << MSR_ME)   |

        ((target_ulong)msr_fe0  << MSR_FE0)  |

        ((target_ulong)msr_se   << MSR_SE)   | /* SE / DWE / UBLE */

        ((target_ulong)msr_be   << MSR_BE)   | /* BE / DE */

        ((target_ulong)msr_fe1  << MSR_FE1)  |

        ((target_ulong)msr_al   << MSR_AL)   |

        ((target_ulong)msr_ip   << MSR_IP)   |

        ((target_ulong)msr_ir   << MSR_IR)   | /* IR / IS */

        ((target_ulong)msr_dr   << MSR_DR)   | /* DR / DS */

        ((target_ulong)msr_pe   << MSR_PE)   | /* PE / EP */

        ((target_ulong)msr_px   << MSR_PX)   | /* PX / PMM */

        ((target_ulong)msr_ri   << MSR_RI)   |

        ((target_ulong)msr_le   << MSR_LE);

}
