static uint64_t build_channel_report_mcic(void)

{

    uint64_t mcic;



    /* subclass: indicate channel report pending */

    mcic = MCIC_SC_CP |

    /* subclass modifiers: none */

    /* storage errors: none */

    /* validity bits: no damage */

        MCIC_VB_WP | MCIC_VB_MS | MCIC_VB_PM | MCIC_VB_IA | MCIC_VB_FP |

        MCIC_VB_GR | MCIC_VB_CR | MCIC_VB_ST | MCIC_VB_AR | MCIC_VB_PR |

        MCIC_VB_FC | MCIC_VB_CT | MCIC_VB_CC;

    if (s390_has_feat(S390_FEAT_VECTOR)) {

        mcic |= MCIC_VB_VR;

    }

    if (s390_has_feat(S390_FEAT_GUARDED_STORAGE)) {

        mcic |= MCIC_VB_GS;

    }

    return mcic;

}
