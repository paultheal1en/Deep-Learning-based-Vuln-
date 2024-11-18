static int kvm_put_sregs(CPUState *env)

{

    struct kvm_sregs sregs;



    memset(sregs.interrupt_bitmap, 0, sizeof(sregs.interrupt_bitmap));

    if (env->interrupt_injected >= 0) {

        sregs.interrupt_bitmap[env->interrupt_injected / 64] |=

                (uint64_t)1 << (env->interrupt_injected % 64);

    }



    if ((env->eflags & VM_MASK)) {

        set_v8086_seg(&sregs.cs, &env->segs[R_CS]);

        set_v8086_seg(&sregs.ds, &env->segs[R_DS]);

        set_v8086_seg(&sregs.es, &env->segs[R_ES]);

        set_v8086_seg(&sregs.fs, &env->segs[R_FS]);

        set_v8086_seg(&sregs.gs, &env->segs[R_GS]);

        set_v8086_seg(&sregs.ss, &env->segs[R_SS]);

    } else {

        set_seg(&sregs.cs, &env->segs[R_CS]);

        set_seg(&sregs.ds, &env->segs[R_DS]);

        set_seg(&sregs.es, &env->segs[R_ES]);

        set_seg(&sregs.fs, &env->segs[R_FS]);

        set_seg(&sregs.gs, &env->segs[R_GS]);

        set_seg(&sregs.ss, &env->segs[R_SS]);

    }



    set_seg(&sregs.tr, &env->tr);

    set_seg(&sregs.ldt, &env->ldt);



    sregs.idt.limit = env->idt.limit;

    sregs.idt.base = env->idt.base;


    sregs.gdt.limit = env->gdt.limit;

    sregs.gdt.base = env->gdt.base;




    sregs.cr0 = env->cr[0];

    sregs.cr2 = env->cr[2];

    sregs.cr3 = env->cr[3];

    sregs.cr4 = env->cr[4];



    sregs.cr8 = cpu_get_apic_tpr(env->apic_state);

    sregs.apic_base = cpu_get_apic_base(env->apic_state);



    sregs.efer = env->efer;



    return kvm_vcpu_ioctl(env, KVM_SET_SREGS, &sregs);

}