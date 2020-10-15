import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.learning.ExpectationMaximization;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import ch.idsia.crema.utility.RandomUtil;
import gnu.trove.map.TIntIntMap;

import java.io.IOException;

public class LogLexample7D {
    public static void main(String[] args) throws IOException, InterruptedException {

        int N = 2000;

/*
    StructuralCausalModel model = new StructuralCausalModel();
    int u = model.addVariable(7, true);
    int x1 = model.addVariable(2);

    model.addParents(x1, u);



    model.setFactor(u, new BayesianFactor(model.getDomain(u), new double[]{0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1 }));
    model.setFactor(x1, EquationBuilder.of(model).fromVector(x1, 0,1,1,1,0,1,0));

    model.toVCredal(model.getEmpiricalNet().getFactors());

*/

    /*
    bug???*/
        StructuralCausalModel model = new StructuralCausalModel();
        int u = model.addVariable(7, true);
        int x1 = model.addVariable(2);
        int x2 = model.addVariable(2);

        model.addParents(x1, u);
        model.addParents(x2,x1,u);




        model.setFactor(u, new BayesianFactor(model.getDomain(u), new double[]{0.2, 0.05, 0.1, 0.25, 0.1, 0.1, 0.2 }));
        model.setFactor(x1, EquationBuilder.of(model).fromVector(x1, 0,1,1,1,0,1,0));
        model.setFactor(x2, EquationBuilder.of(model).fromVector(x2, 0,1,0,0,0,0,1,  0,1,1,1,0,0,1));


        System.out.println(model.toVCredal(BayesianFactor.combineAll(model.getEmpiricalNet().getFactors())));


        TIntIntMap[] data = model.getEmpiricalNet().samples(N);


        // randomize P(U)
        StructuralCausalModel rmodel = model.copy();
        double[] initPoint;
        initPoint = new double[]{0.7, 0.05,0.05, 0.05,0.05, 0.05,0.05};

        initPoint = new double[]{0.7, 0.04,0.01, 0.05,0.05, 0.05,0.05};
        initPoint = new double[]{0.01, 0.01, 0.01, 0.4-0.03, 0.3, 0.1, 0.2};


        rmodel.setFactor(u, new BayesianFactor(model.getDomain(u), initPoint));


        // Run EM in the causal model
        ExpectationMaximization em =
                new ExpectationMaximization(rmodel)
                        .setVerbose(false)
                        .setRecordIntermediate(true)
                        .setRegularization(0.0)
                        .setTrainableVars(model.getExogenousVars());


        // run the method
        em.run(data, 10);

        // Extract the learnt model
        StructuralCausalModel postModel = (StructuralCausalModel) em.getPosterior();
        System.out.println(postModel);



        em.getIntermediateModels().stream().forEach(m -> System.out.println(m.getFactor(u)));


    }
}
