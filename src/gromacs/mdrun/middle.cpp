/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2011,2012,2013,2014,2015,2016,2017,2018, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief Implements the integrator for "middle" thermostating scheme
 *
 * \author Zhi Zi <zhizi@pku.edu.cn>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <memory>

#include "gromacs/awh/awh.h"
#include "gromacs/commandline/filenm.h"
#include "gromacs/compat/make_unique.h"
#include "gromacs/domdec/collect.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_network.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/domdec/partition.h"
#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme-load-balancing.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/imd/imd.h"
#include "gromacs/listed-forces/manage-threading.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdlib/checkpointhandler.h"
#include "gromacs/mdlib/compute_io.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/ebin.h"
#include "gromacs/mdlib/expanded.h"
#include "gromacs/mdlib/force.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/forcerec.h"
#include "gromacs/mdlib/md_support.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/mdebin.h"
#include "gromacs/mdlib/mdoutf.h"
#include "gromacs/mdlib/mdrun.h"
#include "gromacs/mdlib/mdsetup.h"
#include "gromacs/mdlib/membed.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/mdlib/nbnxn_gpu_data_mgmt.h"
#include "gromacs/mdlib/ns.h"
#include "gromacs/mdlib/resethandler.h"
#include "gromacs/mdlib/shellfc.h"
#include "gromacs/mdlib/sighandler.h"
#include "gromacs/mdlib/sim_util.h"
#include "gromacs/mdlib/simulationsignal.h"
#include "gromacs/mdlib/stophandler.h"
#include "gromacs/mdlib/tgroup.h"
#include "gromacs/mdlib/trajectory_writing.h"
#include "gromacs/mdlib/update.h"
#include "gromacs/mdlib/vcm.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/awh-history.h"
#include "gromacs/mdtypes/awh-params.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/df_history.h"
#include "gromacs/mdtypes/energyhistory.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/pullhistory.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/mshift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/output.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/swap/swapcoords.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/mtop_util.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"

#include "integrator.h"
#include "replicaexchange.h"

#if GMX_FAHCORE
#include "corewrap.h"
#endif

using gmx::SimulationSignaller;



void gmx::Integrator::do_middle()
{
    t_inputrec              *ir   = inputrec;
    gmx_mdoutf              *outf = nullptr;
    int64_t                  step, step_rel;
    double                   t, t0, lam0[efptNR];
    bool                     isLastStep               = false;
    bool                     doFreeEnergyPerturbation = false;
    int                      force_flags, cglo_flags;
    tensor                   force_vir, shake_vir, total_vir, pres;
    rvec                     mu_tot;
    gmx_localtop_t          *top;
    t_mdebin                *mdebin   = nullptr;
    gmx_enerdata_t          *enerd;
    PaddedVector<gmx::RVec>  f {};
    gmx_global_stat_t        gstat;
    gmx_update_t            *upd   = nullptr;
    t_graph                 *graph = nullptr;
    gmx_groups_t            *groups;
    gmx_ekindata_t          *ekind;
    gmx_shellfc_t           *shellfc;
    gmx_bool                 bSimAnn = FALSE;
    t_vcm                   *vcm;

    gmx_bool                 do_log, do_ene, do_dr, do_or;

    gmx_bool                 bTemp, bPres, bTrotter;
    int                    **trotter_seq;
    double                   cycles;
    t_extmass                MassQ;

    /* Domain decomposition could incorrectly miss a bonded
       interaction, but checking for that requires a global
       communication stage, which does not otherwise happen in DD
       code. So we do that alongside the first global energy reduction
       after a new DD is made. These variables handle whether the
       check happens, and the result it returns. */
    bool              shouldCheckNumberOfBondedInteractions = false;
    int               totalNumberOfBondedInteractions       = -1;

    SimulationSignals signals;
    // Most global communnication stages don't propagate mdrun
    // signals, and will use this object to achieve that.
    SimulationSignaller nullSignaller(nullptr, nullptr, nullptr, false, false);

    if (ir->bExpanded)
    {
        gmx_fatal(FARGS, "Expanded ensemble is not available in middle scheme.");
    }
    if (ir->bSimTemp)
    {
        gmx_fatal(FARGS, "Simulated tempering is not available in middle scheme.");
    }
    if (ir->bDoAwh)
    {
        gmx_fatal(FARGS, "AWH is not available in middle scheme.");
    }
    if (replExParams.exchangeInterval > 0)
    {
        gmx_fatal(FARGS, "Replica exchange is not available in middle scheme.");
    }
    if (opt2bSet("-ei", nfile, fnm) || observablesHistory->edsamHistory != nullptr)
    {
        gmx_fatal(FARGS, "Essential dynamics is not available in middle scheme.");
    }
    if (ir->bIMD)
    {
        gmx_fatal(FARGS, "Interactive MD is not available in middle scheme.");
    }
    if (isMultiSim(ms))
    {
        gmx_fatal(FARGS, "Multiple simulations is not available in middle scheme.");
    }
    if (std::any_of(ir->opts.annealing, ir->opts.annealing + ir->opts.ngtc,
                    [](int i){return i != eannNO; }))
    {
        gmx_fatal(FARGS, "Simulated annealing is not available in middle scheme.");
    }

    /* Settings for rerun */
    ir->nstlist       = 1;
    ir->nstcalcenergy = 1;
    int        nstglobalcomm = 1;
    const bool bNS           = true;


    ir->nstxout_compressed                   = 0;
    groups                                   = &top_global->groups;


    /* ########### BEGIN: initialize data ########### */ 

    /* Initial values */
    
    init_md(fplog, cr, outputProvider, ir, oenv, mdrunOptions,
            &t, &t0, state_global, lam0,
            nrnb, top_global, &upd, deform,
            nfile, fnm, &outf, &mdebin,
            force_vir, shake_vir, total_vir, pres, mu_tot, &bSimAnn, &vcm, wcycle);

    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda,
                  enerd);

    /* Kinetic energy data */
    snew(ekind, 1);
    init_ekindata(fplog, top_global, &(ir->opts), ekind);
    /* Copy the cos acceleration to the groups struct */
    ekind->cosacc.cos_accel = ir->cos_accel;

    gstat = global_stat_init(ir);

    /* Check for polarizable models and flexible constraints */
    shellfc = init_shell_flexcon(fplog,
                                 top_global, constr ? constr->numFlexibleConstraints() : 0,
                                 ir->nstcalcenergy, DOMAINDECOMP(cr));

    {
        double io = compute_io(ir, top_global->natoms, groups, mdebin->ebin->nener, 1);
        if ((io > 2000) && MASTER(cr))
        {
            fprintf(stderr,
                    "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
                    io);
        }
    }

    // Local state only becomes valid now.
    std::unique_ptr<t_state> stateInstance;
    t_state *                state;

    if (DOMAINDECOMP(cr))
    {
        top = dd_init_local_top(top_global);

        stateInstance = compat::make_unique<t_state>();
        state         = stateInstance.get();
        dd_init_local_state(cr->dd, state_global, state);

        /* Distribute the charge groups over the nodes from the master node */
        dd_partition_system(fplog, mdlog, ir->init_step, cr, TRUE, 1,
                            state_global, top_global, ir,
                            state, &f, mdAtoms, top, fr,
                            vsite, constr,
                            nrnb, nullptr, FALSE);
        shouldCheckNumberOfBondedInteractions = true;
        gmx_bcast(sizeof(ir->nsteps), &ir->nsteps, cr);
    }
    else
    {
        state_change_natoms(state_global, state_global->natoms);
        /* We need to allocate one element extra, since we might use
         * (unaligned) 4-wide SIMD loads to access rvec entries.
         */
        f.resizeWithPadding(state_global->natoms);
        /* Copy the pointer to the global state */
        state = state_global;

        snew(top, 1);
        mdAlgorithmsSetupAtomData(cr, ir, top_global, top, fr,
                                  &graph, mdAtoms, constr, vsite, shellfc);
    }

    auto mdatoms = mdAtoms->mdatoms();

    // NOTE: The global state is no longer used at this point.
    // But state_global is still used as temporary storage space for writing
    // the global state to file and potentially for replica exchange.
    // (Global topology should persist.)

    update_mdatoms(mdatoms, state->lambda[efptMASS]);

    const ContinuationOptions &continuationOptions    = mdrunOptions.continuationOptions;

    /* ########### END: initialize data ########### */ 

    if (ir->efep != efepNO && ir->fepvals->nstdhdl != 0)
    {
        doFreeEnergyPerturbation = true;
    }


    /* ########### BIGIN: calculate globals and remove COM ########### */ 

    /* Be REALLY careful about what flags you set here. You CANNOT assume
     * this is the first step, since we might be restarting from a checkpoint,
     * and in that case we should not do any modifications to the state.
     */
    bool bStopCM = (ir->comm_mode != ecmNO && !ir->bContinuation);

    cglo_flags = (CGLO_INITIALIZATION | CGLO_TEMPERATURE | CGLO_GSTAT
                  | (EI_VV(ir->eI) ? CGLO_PRESSURE : 0)
                  | (EI_VV(ir->eI) ? CGLO_CONSTRAINT : 0)
                  | (continuationOptions.haveReadEkin ? CGLO_READEKIN : 0));
    bool bSumEkinhOld = FALSE;
    /* To minimize communication, compute_globals computes the COM velocity
     * and the kinetic energy for the velocities without COM motion removed.
     * Thus to get the kinetic energy without the COM contribution, we need
     * to call compute_globals twice.
     */
    for (int cgloIteration = 0; cgloIteration < (bStopCM ? 2 : 1); cgloIteration++)
    {
        int cglo_flags_iteration = cglo_flags;
        if (bStopCM && cgloIteration == 0)
        {
            cglo_flags_iteration |= CGLO_STOPCM;
            cglo_flags_iteration &= ~CGLO_TEMPERATURE;
        }
        compute_globals(fplog, gstat, cr, ir, fr, ekind, state, mdatoms, nrnb, vcm,
                        nullptr, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                        constr, &nullSignaller, state->box,
                        &totalNumberOfBondedInteractions, &bSumEkinhOld, cglo_flags_iteration
                        | (shouldCheckNumberOfBondedInteractions ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0));
    }
    checkNumberOfBondedInteractions(mdlog, cr, totalNumberOfBondedInteractions,
                                    top_global, top, state,
                                    &shouldCheckNumberOfBondedInteractions);

    /* ########### END: calculate globals and remove COM ########### */ 

    bTrotter = (EI_VV(ir->eI) && (inputrecNptTrotter(ir) || inputrecNphTrotter(ir) || inputrecNvtTrotter(ir)));

    trotter_seq = init_npt_vars(ir, state, &MassQ, bTrotter);

    if (MASTER(cr))
    {
        fprintf(stderr, "starting Middle Scheme MD run '%s'\n\n",
                *(top_global->name));
        if (mdrunOptions.verbose)
        {
            fprintf(stderr, "Calculated time to finish depends on nsteps from "
                    "run input file,\nwhich may not correspond to the time "
                    "needed to process input trajectory.\n\n");
        }
        fprintf(fplog, "\n");
    }

    walltime_accounting_start_time(walltime_accounting);
    wallcycle_start(wcycle, ewcRUN);
    print_start(fplog, cr, walltime_accounting, "mdrun");

    /***********************************************************
     *
     *             Loop over MD steps
     *
     ************************************************************/

    auto stopHandler = stopHandlerBuilder->getStopHandlerMD(
                compat::not_null<SimulationSignal*>(&signals[eglsSTOPCOND]), false,
                MASTER(cr), ir->nstlist, mdrunOptions.reproducible, nstglobalcomm,
                mdrunOptions.maximumHoursToRun, ir->nstlist == 0, fplog, step, bNS, walltime_accounting);

    // we don't do counter resetting in rerun - finish will always be valid
    walltime_accounting_set_valid_finish(walltime_accounting);

    DdOpenBalanceRegionBeforeForceComputation ddOpenBalanceRegion   =
        (DOMAINDECOMP(cr) ?
         DdOpenBalanceRegionBeforeForceComputation::yes :
         DdOpenBalanceRegionBeforeForceComputation::no);
    DdCloseBalanceRegionAfterForceComputation ddCloseBalanceRegion  =
        (DOMAINDECOMP(cr) ?
         DdCloseBalanceRegionAfterForceComputation::yes :
         DdCloseBalanceRegionAfterForceComputation::no);

    step     = ir->init_step;
    step_rel = 0;

    /* and stop now if we should */
    isLastStep = (isLastStep || (ir->nsteps >= 0 && step_rel > ir->nsteps));
    while (!isLastStep)
    {
        isLastStep = (isLastStep || (ir->nsteps >= 0 && step_rel == ir->nsteps));

        wallcycle_start(wcycle, ewcSTEP);

        t         = t0 + step*ir->delta_t;

        /* update, until we need to calculate force. */

        /* ###### BIGIN: update coordinates ###### */ 

        // if (MASTER(cr))
        {
            wallcycle_start(wcycle, ewcUPDATE);

            update_coords_middle_scheme(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                          ekind, upd, emstrtVELOCITY1, cr, constr);

            update_coords_middle_scheme(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                          ekind, upd, emstrtPOSITION1, cr, constr);



            // Temperature coupling goes here
            {
                /* at the start of step, randomize or scale the velocities ((if vv. Restriction of Andersen controlled
                in preprocessing */

                if (ETC_ANDERSEN(ir->etc)) /* keep this outside of update_tcouple because of the extra info required to pass */
                {
                    gmx_bool bIfRandomize;
                    bIfRandomize = update_randomize_velocities(ir, step, cr, mdatoms, state->v, upd, constr);
                    // /* if we have constraints, we have to remove the kinetic energy parallel to the bonds */
                    // if (constr && bIfRandomize)
                    // {
                    //     constrain_velocities(step, nullptr,
                    //                         state,
                    //                         tmp_vir,
                    //                         constr,
                    //                         bCalcVir, do_log, do_ene);
                    // }
                }

                /* UPDATE PRESSURE VARIABLES IN TROTTER FORMULATION WITH CONSTRAINTS */
                if (bTrotter)
                {
                    // NHC chain is not implemented yet
                    GMX_THROW(NotImplementedError("NHC is not implemented yet in middle scheme"));
                    trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ3);
                    /* We can only do Berendsen coupling after we have summed
                    * the kinetic energy or virial. Since the happens
                    * in global_state after update, we should only do it at
                    * step % nstlist = 1 with bGStatEveryStep=FALSE.
                    */
                }
                else
                {
                    update_tcouple(step, ir, state, ekind, &MassQ, mdatoms);
                }

            }


            update_coords_middle_scheme(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                          ekind, upd, emstrtPOSITION2, cr, constr);

            wallcycle_stop(wcycle, ewcUPDATE);
        }
        
        /* ###### END: update coordinates ###### */ 

        /* In Middle Scheme, the last operator in a step is updating velocities again, 
         * but now the positions have changed.
         * so before that we need to update the force. 
         */
        /* ###### BIGIN: update force ###### */ 

        if (ir->efep != efepNO)
        {
            setCurrentLambdasLocal(step, ir->fepvals, lam0, state);
        }

        if (MASTER(cr))
        {
            const bool constructVsites = ((vsite != nullptr) && mdrunOptions.rerunConstructVsites);
            if (constructVsites && DOMAINDECOMP(cr))
            {
                gmx_fatal(FARGS, "Vsite recalculation with -rerun is not implemented with domain decomposition, "
                          "use a single rank");
            }
        }

        if (DOMAINDECOMP(cr))
        {
            /* Repartition the domain decomposition */
            const bool bMasterState = true;
            dd_partition_system(fplog, mdlog, step, cr,
                                bMasterState, nstglobalcomm,
                                state_global, top_global, ir,
                                state, &f, mdAtoms, top, fr,
                                vsite, constr,
                                nrnb, wcycle,
                                mdrunOptions.verbose);
            shouldCheckNumberOfBondedInteractions = true;
        }

        if (MASTER(cr) && do_log)
        {
            print_ebin_header(fplog, step, t); /* can we improve the information printed here? */
        }

        if (ir->efep != efepNO)
        {
            update_mdatoms(mdatoms, state->lambda[efptMASS]);
        }

        force_flags =   ( GMX_FORCE_STATECHANGED 
                        | ((inputrecDynamicBox(ir)) ? GMX_FORCE_DYNAMICBOX : 0) 
                        | GMX_FORCE_ALLFORCES 
                        | GMX_FORCE_VIRIAL // TODO: Get rid of this once #2649 is solved
                        | GMX_FORCE_ENERGY 
                        | (doFreeEnergyPerturbation ? GMX_FORCE_DHDL : 0)
                        );

        if (shellfc)
        {
            /* Now is the time to relax the shells */
            relax_shell_flexcon(fplog, cr, ms, mdrunOptions.verbose,
                                enforcedRotation, step,
                                ir, bNS, force_flags, top,
                                constr, enerd, fcd,
                                state, f.arrayRefWithPadding(), force_vir, mdatoms,
                                nrnb, wcycle, graph, groups,
                                shellfc, fr, t, mu_tot,
                                vsite,
                                ddOpenBalanceRegion, ddCloseBalanceRegion);
        }
        else
        {
            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too.
             * Check comments in sim_util.c
             */

            /* Since we have no plan to support awh at this moment, we simply 
             * disable it. What's more, if the user has specified to use awh,
             * he should have already recieved a notification at the begining 
             * of this function.
             */
            Awh       *awh = nullptr;
            gmx_edsam *ed  = nullptr;
            do_force(fplog, cr, ms, ir, awh, enforcedRotation,
                     step, nrnb, wcycle, top, groups,
                     state->box, state->x.arrayRefWithPadding(), &state->hist,
                     f.arrayRefWithPadding(), force_vir, mdatoms, enerd, fcd,
                     state->lambda, graph,
                     fr, vsite, mu_tot, t, ed,
                     GMX_FORCE_NS | force_flags,
                     ddOpenBalanceRegion, ddCloseBalanceRegion);
        }
        /* ###### END: update force ###### */ 

        // update velocities after updating force.

        // if (MASTER(cr))
        {
            wallcycle_start(wcycle, ewcUPDATE);
            update_coords_middle_scheme(step, ir, mdatoms, state, f.arrayRefWithPadding(), fcd,
                      ekind, upd, emstrtVELOCITY2, cr, constr);
            wallcycle_stop(wcycle, ewcUPDATE);
        }

        /* Now we have the energies and forces corresponding to the
         * coordinates at time t.
         * So we can write them to the trajectory file now.
         */
        {
            const bool isCheckpointingStep = false;
            const bool doRerun             = false;
            const bool bSumEkinhOld        = false;
            do_md_trajectory_writing(fplog, cr, nfile, fnm, step, step_rel, t,
                                     ir, state, state_global, observablesHistory,
                                     top_global, fr,
                                     outf, mdebin, ekind, f,
                                     isCheckpointingStep, doRerun, isLastStep,
                                     mdrunOptions.writeConfout,
                                     bSumEkinhOld);
        }

        stopHandler->setSignal();

        if (graph)
        {
            /* Need to unshift here */
            unshift_self(graph, state->box, as_rvec_array(state->x.data()));
        }

        if (vsite != nullptr)
        {
            wallcycle_start(wcycle, ewcVSITECONSTR);
            if (graph != nullptr)
            {
                shift_self(graph, state->box, as_rvec_array(state->x.data()));
            }
            construct_vsites(vsite, as_rvec_array(state->x.data()), ir->delta_t, as_rvec_array(state->v.data()),
                             top->idef.iparams, top->idef.il,
                             fr->ePBC, fr->bMolPBC, cr, state->box);

            if (graph != nullptr)
            {
                unshift_self(graph, state->box, as_rvec_array(state->x.data()));
            }
            wallcycle_stop(wcycle, ewcVSITECONSTR);
        }

    /* ########### BIGIN: calculate globals ########### */ 

        {
            const bool          doInterSimSignal = false;
            const bool          doIntraSimSignal = true;
            bool                bSumEkinhOld     = false;
            t_vcm              *vcm              = nullptr;
            SimulationSignaller signaller(&signals, cr, ms, doInterSimSignal, doIntraSimSignal);

            int cglo_flags =    (CGLO_GSTAT 
                                | CGLO_ENERGY
                                | (shouldCheckNumberOfBondedInteractions ? CGLO_CHECK_NUMBER_OF_BONDED_INTERACTIONS : 0)
                                | CGLO_TEMPERATURE
                                // | CGLO_PRESSURE    // stuck forever... need more investigation
                                // | CGLO_CONSTRAINT  // segment fault 11... need more investigation
                                );

            compute_globals(fplog, gstat, cr, ir, fr, ekind, state, mdatoms, nrnb, vcm,
                            wcycle, enerd, nullptr, nullptr, nullptr, nullptr, mu_tot,
                            constr, &signaller,
                            state->box,
                            &totalNumberOfBondedInteractions, &bSumEkinhOld, cglo_flags);
            checkNumberOfBondedInteractions(mdlog, cr, totalNumberOfBondedInteractions,
                                            top_global, top, state,
                                            &shouldCheckNumberOfBondedInteractions);
        }

    /* ########### END: calculate globals ########### */ 

        {
            gmx::HostVector<gmx::RVec>     fglobal(top_global->natoms);
            gmx::ArrayRef<gmx::RVec>       ftemp;
            gmx::ArrayRef<const gmx::RVec> flocal = gmx::makeArrayRef(f);
            if (DOMAINDECOMP(cr))
            {
                ftemp = gmx::makeArrayRef(fglobal);
                dd_collect_vec(cr->dd, state, flocal, ftemp);
            }
            else
            {
                ftemp = gmx::makeArrayRef(f);
            }

        }

        /* Note: this is OK, but there are some numerical precision issues with using the convergence of
           the virial that should probably be addressed eventually. state->veta has better properies,
           but what we actually need entering the new cycle is the new shake_vir value. Ideally, we could
           generate the new shake_vir, but test the veta value for convergence.  This will take some thought. */

        if (ir->efep != efepNO)
        {
            /* Sum up the foreign energy and dhdl terms for md and sd.
               Currently done every step so that dhdl is correct in the .edr */
            sum_dhdl(enerd, state->lambda, ir->fepvals);
        }

        // if (bCalcEner)
        {
            /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */

            /* use the directly determined last velocity, not actually the averaged half steps */
            // if (bTrotter && ir->eI == eiVV)
            // {
            //     enerd->term[F_EKIN] = last_ekin;
            // }
            enerd->term[F_ETOT] = enerd->term[F_EPOT] + enerd->term[F_EKIN];

            // if (integratorHasConservedEnergyQuantity(ir))
            // {
            //     if (EI_VV(ir->eI))
            //     {
            //         enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + saved_conserved_quantity;
            //     }
            //     else
            //     {
            //         enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + NPT_energy(ir, state, &MassQ);
            //     }
            // }
            /* #########  END PREPARING EDR OUTPUT  ###########  */
        }


        /* Output stuff */
        if (MASTER(cr))
        {
            const bool bCalcEnerStep = true;
            upd_mdebin(mdebin, doFreeEnergyPerturbation, bCalcEnerStep,
                       t, mdatoms->tmass, enerd, state,
                       ir->fepvals, ir->expandedvals, state->box,
                       shake_vir, force_vir, total_vir, pres,
                       ekind, mu_tot, constr);

            Awh       *awh    = nullptr;
            do_log     = do_per_step(step, ir->nstlog) || isLastStep;
            do_ene     = do_per_step(step, ir->nstenergy);
            do_dr      = do_per_step(step, ir->nstdisreout);
            do_or      = do_per_step(step, ir->nstorireout);

            print_ebin(mdoutf_get_fp_ene(outf), do_ene, do_dr, do_or, do_log ? fplog : nullptr,
                       step, t,
                       eprNORMAL, mdebin, fcd, groups, &(ir->opts), awh);

            if (do_per_step(step, ir->nstlog))
            {
                if (fflush(fplog) != 0)
                {
                    gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
                }
            }
        }

        /* Print the remaining wall clock time for the run */
        if (isMasterSimMasterRank(ms, cr) &&
            (mdrunOptions.verbose || gmx_got_usr_signal()))
        {
            if (shellfc)
            {
                fprintf(stderr, "\n");
            }
            print_time(stderr, walltime_accounting, step, ir, cr);
        }

        cycles = wallcycle_stop(wcycle, ewcSTEP);
        if (DOMAINDECOMP(cr) && wcycle)
        {
            dd_cycles_add(cr->dd, cycles, ddCyclStep);
        }

        /* increase the MD step number */
        step++;
        step_rel++;
    }
    /* End of main MD loop */

    /* Closing TNG files can include compressing data. Therefore it is good to do that
     * before stopping the time measurements. */
    mdoutf_tng_close(outf);

    /* Stop measuring walltime */
    walltime_accounting_end_time(walltime_accounting);


    if (!thisRankHasDuty(cr, DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    done_mdebin(mdebin);
    done_mdoutf(outf);

    done_shellfc(fplog, shellfc, step_rel);

    // Clean up swapcoords
    if (ir->eSwapCoords != eswapNO)
    {
        finish_swapcoords(ir->swap);
    }

    walltime_accounting_set_nsteps_done(walltime_accounting, step_rel);

    destroy_enerdata(enerd);
    sfree(enerd);
    sfree(top);
}
