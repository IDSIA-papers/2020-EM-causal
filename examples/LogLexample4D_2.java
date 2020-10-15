import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.predefined.RandomChainNonMarkovian;
import ch.idsia.credici.model.predefined.RandomSquares;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.learning.ExpectationMaximization;
import ch.idsia.crema.utility.RandomUtil;
import gnu.trove.map.TIntIntMap;

import java.io.IOException;

public class LogLexample4D_2 {
    public static void main(String[] args) throws IOException, InterruptedException {

        int N = 500;

        StructuralCausalModel model;
        //model = RandomSquares.buildModel(false,2, 2, 7);
        model = RandomChainNonMarkovian.buildModel(4, 2, 5);

        model.fillExogenousWithRandomFactors(4);
        model.fillWithRandomEquations();

        /*
        int nvert;
        do {
            try {
                model.fillExogenousWithRandomFactors(4);
                model.fillWithRandomEquations();
                System.out.print(".");
                VertexFactor f = ((VertexFactor) model.toVCredal(model.getEmpiricalProbs()).getFactor(5));
                System.out.println(f);
                nvert = f.getData()[0].length;
            }catch (Exception e ){
                nvert = 0;
            }
        }
        while(nvert != 1);*/

        int X[] = model.getEndogenousVars();
        int U[] = model.getExogenousVars();

        System.out.println(model.toVCredal(model.getEmpiricalProbs()));


        TIntIntMap[] data = model.samples(N, model.getEndogenousVars());


        // randomize P(U)
        StructuralCausalModel rmodel = model.copy();
        rmodel.fillExogenousWithRandomFactors(4);
        rmodel.fillExogenousWithRandomFactors(2);


        if(false) {
            double[][] initPoint = new double[2][];
            initPoint[0] = new double[]{0.2, 0.3, 0.05, 0.4, 0.05};
            initPoint[1] = new double[]{0.3, 0.05, 0.3, 0.3, 0.05};
            //initPoint[1] = new double[]{0.0392, 0.6772, 0.0278, 0.0668, 0.189};


            rmodel.setFactor(U[0], new BayesianFactor(model.getDomain(U[0]), initPoint[0]));
            rmodel.setFactor(U[1], new BayesianFactor(model.getDomain(U[1]), initPoint[1]));
        }

        // Run EM in the causal model
        ExpectationMaximization em =
                new ExpectationMaximization(rmodel)
                        .setVerbose(false)
                        .setInline(false)
                        .setRecordIntermediate(true)
                        .setRegularization(0.000000001);
                        //.setTrainableVars(model.getExogenousVars());


        // run the method
        em.run(data, 10);

        // Extract the learnt model
        StructuralCausalModel postModel = (StructuralCausalModel) em.getPosterior();
        System.out.println(postModel);


        for(int u : U) {
            System.out.println(u);
            em.getIntermediateModels().stream().forEach(m -> System.out.println(m.getFactor(u)));
        }


    }
}
