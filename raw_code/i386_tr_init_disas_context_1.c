static int i386_tr_init_disas_context(DisasContextBase *dcbase, CPUState *cpu,

                                      int max_insns)

{

    DisasContext *dc = container_of(dcbase, DisasContext, base);

    CPUX86State *env = cpu->env_ptr;

    uint32_t flags = dc->base.tb->flags;

    target_ulong cs_base = dc->base.tb->cs_base;



    dc->pe = (flags >> HF_PE_SHIFT) & 1;

    dc->code32 = (flags >> HF_CS32_SHIFT) & 1;

    dc->ss32 = (flags >> HF_SS32_SHIFT) & 1;

    dc->addseg = (flags >> HF_ADDSEG_SHIFT) & 1;

    dc->f_st = 0;

    dc->vm86 = (flags >> VM_SHIFT) & 1;

    dc->cpl = (flags >> HF_CPL_SHIFT) & 3;

    dc->iopl = (flags >> IOPL_SHIFT) & 3;

    dc->tf = (flags >> TF_SHIFT) & 1;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->cc_op_dirty = false;

    dc->cs_base = cs_base;

    dc->popl_esp_hack = 0;

    /* select memory access functions */

    dc->mem_index = 0;

#ifdef CONFIG_SOFTMMU

    dc->mem_index = cpu_mmu_index(env, false);

#endif

    dc->cpuid_features = env->features[FEAT_1_EDX];

    dc->cpuid_ext_features = env->features[FEAT_1_ECX];

    dc->cpuid_ext2_features = env->features[FEAT_8000_0001_EDX];

    dc->cpuid_ext3_features = env->features[FEAT_8000_0001_ECX];

    dc->cpuid_7_0_ebx_features = env->features[FEAT_7_0_EBX];

    dc->cpuid_xsave_features = env->features[FEAT_XSAVE];

#ifdef TARGET_X86_64

    dc->lma = (flags >> HF_LMA_SHIFT) & 1;

    dc->code64 = (flags >> HF_CS64_SHIFT) & 1;

#endif

    dc->flags = flags;

    dc->jmp_opt = !(dc->tf || dc->base.singlestep_enabled ||

                    (flags & HF_INHIBIT_IRQ_MASK));

    /* Do not optimize repz jumps at all in icount mode, because

       rep movsS instructions are execured with different paths

       in !repz_opt and repz_opt modes. The first one was used

       always except single step mode. And this setting

       disables jumps optimization and control paths become

       equivalent in run and single step modes.

       Now there will be no jump optimization for repz in

       record/replay modes and there will always be an

       additional step for ecx=0 when icount is enabled.

     */

    dc->repz_opt = !dc->jmp_opt && !(dc->base.tb->cflags & CF_USE_ICOUNT);

#if 0

    /* check addseg logic */

    if (!dc->addseg && (dc->vm86 || !dc->pe || !dc->code32))

        printf("ERROR addseg\n");

#endif



    cpu_T0 = tcg_temp_new();

    cpu_T1 = tcg_temp_new();

    cpu_A0 = tcg_temp_new();



    cpu_tmp0 = tcg_temp_new();

    cpu_tmp1_i64 = tcg_temp_new_i64();

    cpu_tmp2_i32 = tcg_temp_new_i32();

    cpu_tmp3_i32 = tcg_temp_new_i32();

    cpu_tmp4 = tcg_temp_new();

    cpu_ptr0 = tcg_temp_new_ptr();

    cpu_ptr1 = tcg_temp_new_ptr();

    cpu_cc_srcT = tcg_temp_local_new();



    return max_insns;

}
