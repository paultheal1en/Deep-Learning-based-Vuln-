build_hash_table (const sparc_opcode **opcode_table,

                  sparc_opcode_hash **hash_table,

                  int num_opcodes)

{

  int i;

  int hash_count[HASH_SIZE];

  static sparc_opcode_hash *hash_buf = NULL;



  /* Start at the end of the table and work backwards so that each

     chain is sorted.  */



  memset (hash_table, 0, HASH_SIZE * sizeof (hash_table[0]));

  memset (hash_count, 0, HASH_SIZE * sizeof (hash_count[0]));

  if (hash_buf != NULL)

    free (hash_buf);

  hash_buf = malloc (sizeof (* hash_buf) * num_opcodes);

  for (i = num_opcodes - 1; i >= 0; --i)

    {

      int hash = HASH_INSN (opcode_table[i]->match);

      sparc_opcode_hash *h = &hash_buf[i];



      h->next = hash_table[hash];

      h->opcode = opcode_table[i];

      hash_table[hash] = h;

      ++hash_count[hash];

    }



#if 0 /* for debugging */

  {

    int min_count = num_opcodes, max_count = 0;

    int total;



    for (i = 0; i < HASH_SIZE; ++i)

      {

        if (hash_count[i] < min_count)

          min_count = hash_count[i];

        if (hash_count[i] > max_count)

          max_count = hash_count[i];

        total += hash_count[i];

      }



    printf ("Opcode hash table stats: min %d, max %d, ave %f\n",

            min_count, max_count, (double) total / HASH_SIZE);

  }

#endif

}
