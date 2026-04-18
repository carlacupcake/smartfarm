# Response to Reviewer Comments

---

## Major Comments

**1. "The GA optimizes four scalar parameters while MPC solves a daily continuous-valued problem, and the two methods use different objective formulations (linear costs in equations 51 to 53 vs. quadratic penalties in Eq. 59). The paper's central conclusion rests on this comparison, so the authors should explicitly discuss how these structural and formulation asymmetries may influence the outcome, and whether the results reflect a fundamental property of open-loop vs. closed-loop control or partly an artifact of the comparison design, as the computational complexity of the two methods is very different."**

We added a new subsection to the Discussion (Section 6.2, "Comparison design: structural and formulation asymmetries") that explicitly addresses three sources of asymmetry: (1) the decision-space structure (4-parameter GA vs. daily continuous MPC), (2) the objective formulation (linear revenue vs. quadratic penalties), and (3) computational effort. The subsection discusses how the GA's low-dimensional parameterization acts as implicit regularization, how MPC's quadratic penalties impose an implicit resource conservation preference that partly explains its lower revenue but dramatically lower resource usage, and synthesizes what is fundamental to the system dynamics versus what is a design choice. A forward reference was added to the existing interpretation paragraph (Section 6.1), and the Conclusion was expanded to acknowledge these asymmetries alongside the horizon-based explanation.

**2. "MPC uses dramatically less water and fertilizer yet achieves lower revenue. It would strengthen the paper to disentangle whether this stems from the short planning horizon or because of resource conservation. For instance, by running a resource-constrained GA with total seasonal inputs capped at MPC's usage levels."**

We implemented the reviewer's suggested ablation study. We added optional resource constraint parameters to the GA implementation and ran a resource-constrained GA with total seasonal inputs capped at MPC's maximum usage levels (2.0 inches water, 310 lbs fertilizer). The constrained GA achieved a mean revenue of $759/acre, compared with MPC's $750/acre, winning 11 of 21 scenarios. This near-parity (vs. the unconstrained GA's $863/acre) indicates that the majority of the revenue gap is attributable to the implicit resource conservation in MPC's quadratic formulation, not horizon myopia alone. These results are presented in the "Fundamental versus artifact" paragraph of Section 6.2. The experiment notebook is provided at `src/smartfarm/examples/ga/ga_resource_constrained.ipynb`.

---

## Minor Comments

**3. "The MPC assumes perfect weather forecasts, which should favor it. Do a parameter sweep experiment with noisy forecast inputs and see whether the results hold."**

Rather than running a sweep (which would only widen the gap in GA's favor), we added a "Forecast sensitivity" paragraph to Section 6.2 explaining why this experiment is unnecessary: MPC is the only method that consumes forecast information at runtime, so degrading forecast quality can only reduce MPC's performance while leaving the GA unaffected. The GA's robustness to weather uncertainty is already tested by evaluating strategies optimized for one condition across all 21 scenarios. We also expanded the "Perfect forecast assumption" limitation to cross-reference this discussion and frame stochastic/robust MPC as future work.

**4. "The absence of soil moisture dynamics likely favors the GA's infrequent large-volume irrigation strategy; please discuss this potential bias more explicitly."**

We expanded the "Soil moisture dynamics" limitation (Section 6.4) to explicitly note that without soil moisture modeling, all applied water is treated as directly available regardless of application volume. In practice, large single applications would exceed the soil's water-holding capacity, with excess lost to deep drainage or runoff. MPC's frequent, small applications would suffer less from such losses, so incorporating a soil water balance model could narrow the GA--MPC performance gap.

**5. "The additive extremity index assumes non-interacting stressors; note this limitation when interpreting compound-event scenarios."**

We added a clarifying paragraph immediately after the aggregate extremity equation (Section 4.5) noting that the additive form assumes independent stressor contributions. In practice, compound events such as simultaneous drought and heat stress can interact synergistically, so the index should be interpreted as a lower bound on effective stress severity for compound-event scenarios, and revenue declines may be steeper than the index alone would suggest.

**6. "The economic model omits per-event operational costs, which would differentially affect strategies with varying application frequencies."**

We added a "Per-event operational costs" limitation (Section 6.4) noting that the objective function penalizes total resource quantity but not the number of application events. In practice, each event incurs operational costs (labor, equipment, fuel) regardless of amount applied, which would differentially penalize MPC's frequent daily applications more than the GA's infrequent seasonal applications.

**7. "The abstract reports both '35% higher revenue' and a '56% improvement', please clarify what each of those is."**

We clarified the distinction throughout the manuscript. In the abstract, "35%" now explicitly reads "35% higher mean revenue ... across all 21 scenarios." In the results section, the "56% improvement" is now clarified as applying to the baseline weather scenario specifically ("a 56% improvement on that scenario"). The bullet-point summary and conclusion already used clear language ("on average" and "higher mean revenue," respectively).

**8. "Reference [18] has a missing publisher location ('???')."**

We added `address = {Madison, WI}` to the `rawlings2017mpc` entry in `bibliography.bib`. Nob Hill Publishing is based in Madison, Wisconsin.

**9. "Equation 5: clarify behavior of the inverse nu_T and nu_R terms as these factors approach zero."**

We added a paragraph in the Growth Dynamics subsection (Section 2.1), after the presentation of both equations containing inverse nutrient factor terms (Eqs. 5 and 7), explaining that: (1) the nutrient factors are strictly positive (defined as exponentials) but can become arbitrarily small under prolonged stress, (2) as they approach zero the inverse terms grow without bound, which would produce unrealistically large growth rates and carrying capacities, and (3) the implementation clamps all modulated values to physiologically reasonable ranges---growth rates to [0, 2x nominal] and carrying capacities to [x(t), 2x nominal]---with physiological justification for both bounds.

**10. "If time permits, a brief sensitivity analysis on crop-specific parameters or a second crop demonstration would support the generalizability claims in the conclusion, but this should not be a necessary experiment for acceptance of this paper."**

We appreciate this suggestion and view it as a natural next step for future work. The conclusion has been kept appropriately scoped, noting that "the framework generalizes readily to other crops by adjusting the growth parameters and input-effect relationships," which describes the model's structural flexibility without overclaiming empirical validation across crops. We intend to pursue a multi-crop demonstration (soybean and wheat) in a follow-up study.
