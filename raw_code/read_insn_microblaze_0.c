read_insn_microblaze (bfd_vma memaddr, 

		      struct disassemble_info *info,

		      struct op_code_struct **opr)

{

  unsigned char       ibytes[4];

  int                 status;

  struct op_code_struct * op;

  unsigned long inst;



  status = info->read_memory_func (memaddr, ibytes, 4, info);



  if (status != 0) 

    {

      info->memory_error_func (status, memaddr, info);

      return 0;

    }



  if (info->endian == BFD_ENDIAN_BIG)

    inst = (ibytes[0] << 24) | (ibytes[1] << 16) | (ibytes[2] << 8) | ibytes[3];

  else if (info->endian == BFD_ENDIAN_LITTLE)

    inst = (ibytes[3] << 24) | (ibytes[2] << 16) | (ibytes[1] << 8) | ibytes[0];

  else

    abort ();



  /* Just a linear search of the table.  */

  for (op = opcodes; op->name != 0; op ++)

    if (op->bit_sequence == (inst & op->opcode_mask))

      break;



  *opr = op;

  return inst;

}
