build_madt(GArray *table_data, GArray *linker, PCMachineState *pcms)

{

    MachineClass *mc = MACHINE_GET_CLASS(pcms);

    CPUArchIdList *apic_ids = mc->possible_cpu_arch_ids(MACHINE(pcms));

    int madt_start = table_data->len;



    AcpiMultipleApicTable *madt;

    AcpiMadtIoApic *io_apic;

    AcpiMadtIntsrcovr *intsrcovr;

    AcpiMadtLocalNmi *local_nmi;

    int i;



    madt = acpi_data_push(table_data, sizeof *madt);

    madt->local_apic_address = cpu_to_le32(APIC_DEFAULT_ADDRESS);

    madt->flags = cpu_to_le32(1);



    for (i = 0; i < apic_ids->len; i++) {

        AcpiMadtProcessorApic *apic = acpi_data_push(table_data, sizeof *apic);

        int apic_id = apic_ids->cpus[i].arch_id;



        apic->type = ACPI_APIC_PROCESSOR;

        apic->length = sizeof(*apic);

        apic->processor_id = i;

        apic->local_apic_id = apic_id;

        if (apic_ids->cpus[i].cpu != NULL) {

            apic->flags = cpu_to_le32(1);

        } else {

            /* ACPI spec says that LAPIC entry for non present

             * CPU may be omitted from MADT or it must be marked

             * as disabled. However omitting non present CPU from

             * MADT breaks hotplug on linux. So possible CPUs

             * should be put in MADT but kept disabled.

             */

            apic->flags = cpu_to_le32(0);

        }

    }

    g_free(apic_ids);



    io_apic = acpi_data_push(table_data, sizeof *io_apic);

    io_apic->type = ACPI_APIC_IO;

    io_apic->length = sizeof(*io_apic);

#define ACPI_BUILD_IOAPIC_ID 0x0

    io_apic->io_apic_id = ACPI_BUILD_IOAPIC_ID;

    io_apic->address = cpu_to_le32(IO_APIC_DEFAULT_ADDRESS);

    io_apic->interrupt = cpu_to_le32(0);



    if (pcms->apic_xrupt_override) {

        intsrcovr = acpi_data_push(table_data, sizeof *intsrcovr);

        intsrcovr->type   = ACPI_APIC_XRUPT_OVERRIDE;

        intsrcovr->length = sizeof(*intsrcovr);

        intsrcovr->source = 0;

        intsrcovr->gsi    = cpu_to_le32(2);

        intsrcovr->flags  = cpu_to_le16(0); /* conforms to bus specifications */

    }

    for (i = 1; i < 16; i++) {

#define ACPI_BUILD_PCI_IRQS ((1<<5) | (1<<9) | (1<<10) | (1<<11))

        if (!(ACPI_BUILD_PCI_IRQS & (1 << i))) {

            /* No need for a INT source override structure. */

            continue;

        }

        intsrcovr = acpi_data_push(table_data, sizeof *intsrcovr);

        intsrcovr->type   = ACPI_APIC_XRUPT_OVERRIDE;

        intsrcovr->length = sizeof(*intsrcovr);

        intsrcovr->source = i;

        intsrcovr->gsi    = cpu_to_le32(i);

        intsrcovr->flags  = cpu_to_le16(0xd); /* active high, level triggered */

    }



    local_nmi = acpi_data_push(table_data, sizeof *local_nmi);

    local_nmi->type         = ACPI_APIC_LOCAL_NMI;

    local_nmi->length       = sizeof(*local_nmi);

    local_nmi->processor_id = 0xff; /* all processors */

    local_nmi->flags        = cpu_to_le16(0);

    local_nmi->lint         = 1; /* ACPI_LINT1 */



    build_header(linker, table_data,

                 (void *)(table_data->data + madt_start), "APIC",

                 table_data->len - madt_start, 1, NULL, NULL);

}
