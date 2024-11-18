static int kvm_s390_store_adtl_status(S390CPU *cpu, hwaddr addr)

{

    void *mem;

    hwaddr len = ADTL_SAVE_AREA_SIZE;



    mem = cpu_physical_memory_map(addr, &len, 1);

    if (!mem) {

        return -EFAULT;

    }

    if (len != ADTL_SAVE_AREA_SIZE) {

        cpu_physical_memory_unmap(mem, len, 1, 0);

        return -EFAULT;

    }



    memcpy(mem, &cpu->env.vregs, 512);



    cpu_physical_memory_unmap(mem, len, 1, len);



    return 0;

}
