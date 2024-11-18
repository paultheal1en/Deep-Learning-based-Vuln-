static int cpu_post_load(void *opaque, int version_id)

{

    ARMCPU *cpu = opaque;

    int i, v;



    /* Update the values list from the incoming migration data.

     * Anything in the incoming data which we don't know about is

     * a migration failure; anything we know about but the incoming

     * data doesn't specify retains its current (reset) value.

     * The indexes list remains untouched -- we only inspect the

     * incoming migration index list so we can match the values array

     * entries with the right slots in our own values array.

     */



    for (i = 0, v = 0; i < cpu->cpreg_array_len

             && v < cpu->cpreg_vmstate_array_len; i++) {

        if (cpu->cpreg_vmstate_indexes[v] > cpu->cpreg_indexes[i]) {

            /* register in our list but not incoming : skip it */

            continue;

        }

        if (cpu->cpreg_vmstate_indexes[v] < cpu->cpreg_indexes[i]) {

            /* register in their list but not ours: fail migration */

            return -1;

        }

        /* matching register, copy the value over */

        cpu->cpreg_values[i] = cpu->cpreg_vmstate_values[v];

        v++;

    }



    if (kvm_enabled()) {

        if (!write_list_to_kvmstate(cpu)) {

            return -1;

        }

        /* Note that it's OK for the TCG side not to know about

         * every register in the list; KVM is authoritative if

         * we're using it.

         */

        write_list_to_cpustate(cpu);

    } else {

        if (!write_list_to_cpustate(cpu)) {

            return -1;

        }

    }



    hw_breakpoint_update_all(cpu);

    hw_watchpoint_update_all(cpu);



    return 0;

}
