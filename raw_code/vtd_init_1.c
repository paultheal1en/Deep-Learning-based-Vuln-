static void vtd_init(IntelIOMMUState *s)

{

    X86IOMMUState *x86_iommu = X86_IOMMU_DEVICE(s);



    memset(s->csr, 0, DMAR_REG_SIZE);

    memset(s->wmask, 0, DMAR_REG_SIZE);

    memset(s->w1cmask, 0, DMAR_REG_SIZE);

    memset(s->womask, 0, DMAR_REG_SIZE);



    s->iommu_ops.translate = vtd_iommu_translate;

    s->iommu_ops.notify_started = vtd_iommu_notify_started;

    s->root = 0;

    s->root_extended = false;

    s->dmar_enabled = false;

    s->iq_head = 0;

    s->iq_tail = 0;

    s->iq = 0;

    s->iq_size = 0;

    s->qi_enabled = false;

    s->iq_last_desc_type = VTD_INV_DESC_NONE;

    s->next_frcd_reg = 0;

    s->cap = VTD_CAP_FRO | VTD_CAP_NFR | VTD_CAP_ND | VTD_CAP_MGAW |

             VTD_CAP_SAGAW | VTD_CAP_MAMV | VTD_CAP_PSI | VTD_CAP_SLLPS;

    s->ecap = VTD_ECAP_QI | VTD_ECAP_IRO;



    if (x86_iommu->intr_supported) {

        s->ecap |= VTD_ECAP_IR | VTD_ECAP_EIM;

    }



    vtd_reset_context_cache(s);

    vtd_reset_iotlb(s);



    /* Define registers with default values and bit semantics */

    vtd_define_long(s, DMAR_VER_REG, 0x10UL, 0, 0);

    vtd_define_quad(s, DMAR_CAP_REG, s->cap, 0, 0);

    vtd_define_quad(s, DMAR_ECAP_REG, s->ecap, 0, 0);

    vtd_define_long(s, DMAR_GCMD_REG, 0, 0xff800000UL, 0);

    vtd_define_long_wo(s, DMAR_GCMD_REG, 0xff800000UL);

    vtd_define_long(s, DMAR_GSTS_REG, 0, 0, 0);

    vtd_define_quad(s, DMAR_RTADDR_REG, 0, 0xfffffffffffff000ULL, 0);

    vtd_define_quad(s, DMAR_CCMD_REG, 0, 0xe0000003ffffffffULL, 0);

    vtd_define_quad_wo(s, DMAR_CCMD_REG, 0x3ffff0000ULL);



    /* Advanced Fault Logging not supported */

    vtd_define_long(s, DMAR_FSTS_REG, 0, 0, 0x11UL);

    vtd_define_long(s, DMAR_FECTL_REG, 0x80000000UL, 0x80000000UL, 0);

    vtd_define_long(s, DMAR_FEDATA_REG, 0, 0x0000ffffUL, 0);

    vtd_define_long(s, DMAR_FEADDR_REG, 0, 0xfffffffcUL, 0);



    /* Treated as RsvdZ when EIM in ECAP_REG is not supported

     * vtd_define_long(s, DMAR_FEUADDR_REG, 0, 0xffffffffUL, 0);

     */

    vtd_define_long(s, DMAR_FEUADDR_REG, 0, 0, 0);



    /* Treated as RO for implementations that PLMR and PHMR fields reported

     * as Clear in the CAP_REG.

     * vtd_define_long(s, DMAR_PMEN_REG, 0, 0x80000000UL, 0);

     */

    vtd_define_long(s, DMAR_PMEN_REG, 0, 0, 0);



    vtd_define_quad(s, DMAR_IQH_REG, 0, 0, 0);

    vtd_define_quad(s, DMAR_IQT_REG, 0, 0x7fff0ULL, 0);

    vtd_define_quad(s, DMAR_IQA_REG, 0, 0xfffffffffffff007ULL, 0);

    vtd_define_long(s, DMAR_ICS_REG, 0, 0, 0x1UL);

    vtd_define_long(s, DMAR_IECTL_REG, 0x80000000UL, 0x80000000UL, 0);

    vtd_define_long(s, DMAR_IEDATA_REG, 0, 0xffffffffUL, 0);

    vtd_define_long(s, DMAR_IEADDR_REG, 0, 0xfffffffcUL, 0);

    /* Treadted as RsvdZ when EIM in ECAP_REG is not supported */

    vtd_define_long(s, DMAR_IEUADDR_REG, 0, 0, 0);



    /* IOTLB registers */

    vtd_define_quad(s, DMAR_IOTLB_REG, 0, 0Xb003ffff00000000ULL, 0);

    vtd_define_quad(s, DMAR_IVA_REG, 0, 0xfffffffffffff07fULL, 0);

    vtd_define_quad_wo(s, DMAR_IVA_REG, 0xfffffffffffff07fULL);



    /* Fault Recording Registers, 128-bit */

    vtd_define_quad(s, DMAR_FRCD_REG_0_0, 0, 0, 0);

    vtd_define_quad(s, DMAR_FRCD_REG_0_2, 0, 0, 0x8000000000000000ULL);



    /*

     * Interrupt remapping registers.

     */

    vtd_define_quad(s, DMAR_IRTA_REG, 0, 0xfffffffffffff80fULL, 0);

}
