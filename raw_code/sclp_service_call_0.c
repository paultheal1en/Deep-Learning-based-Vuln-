int sclp_service_call(CPUS390XState *env, uint32_t sccb, uint64_t code)

{

    int r = 0;

    int shift = 0;



#ifdef DEBUG_HELPER

    printf("sclp(0x%x, 0x%" PRIx64 ")\n", sccb, code);

#endif



    if (sccb & ~0x7ffffff8ul) {

        fprintf(stderr, "KVM: invalid sccb address 0x%x\n", sccb);

        r = -1;

        goto out;

    }



    switch(code) {

        case SCLP_CMDW_READ_SCP_INFO:

        case SCLP_CMDW_READ_SCP_INFO_FORCED:

            while ((ram_size >> (20 + shift)) > 65535) {

                shift++;

            }

            stw_phys(sccb + SCP_MEM_CODE, ram_size >> (20 + shift));

            stb_phys(sccb + SCP_INCREMENT, 1 << shift);

            stw_phys(sccb + SCP_RESPONSE_CODE, 0x10);



            if (kvm_enabled()) {

#ifdef CONFIG_KVM

                kvm_s390_interrupt_internal(env, KVM_S390_INT_SERVICE,

                                            sccb & ~3, 0, 1);

#endif

            } else {

                env->psw.addr += 4;

                ext_interrupt(env, EXT_SERVICE, sccb & ~3, 0);

            }

            break;

        default:

#ifdef DEBUG_HELPER

            printf("KVM: invalid sclp call 0x%x / 0x%" PRIx64 "x\n", sccb, code);

#endif

            r = -1;

            break;

    }



out:

    return r;

}
