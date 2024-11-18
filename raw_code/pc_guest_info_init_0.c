void pc_guest_info_init(PCMachineState *pcms)

{

    int i, j;



    pcms->apic_xrupt_override = kvm_allows_irq0_override();

    pcms->numa_nodes = nb_numa_nodes;

    pcms->node_mem = g_malloc0(pcms->numa_nodes *

                                    sizeof *pcms->node_mem);

    for (i = 0; i < nb_numa_nodes; i++) {

        pcms->node_mem[i] = numa_info[i].node_mem;

    }



    pcms->node_cpu = g_malloc0(pcms->apic_id_limit *

                                     sizeof *pcms->node_cpu);



    for (i = 0; i < max_cpus; i++) {

        unsigned int apic_id = x86_cpu_apic_id_from_index(i);

        assert(apic_id < pcms->apic_id_limit);

        for (j = 0; j < nb_numa_nodes; j++) {

            if (test_bit(i, numa_info[j].node_cpu)) {

                pcms->node_cpu[apic_id] = j;

                break;

            }

        }

    }



    pcms->machine_done.notify = pc_machine_done;

    qemu_add_machine_init_done_notifier(&pcms->machine_done);

}
