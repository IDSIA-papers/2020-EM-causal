import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.crema.data.WriterCSV;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.learning.ExpectationMaximization;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import com.google.common.primitives.Doubles;
import gnu.trove.map.TIntIntMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LogLexample4D {
    public static void main(String[] args) throws IOException, InterruptedException {

        int N = 10000;

        StructuralCausalModel model = new StructuralCausalModel();
        int u = model.addVariable(4, true);
        int x = model.addVariable(2);
        model.addParents(x, u);


        model.setFactor(u, new BayesianFactor(model.getDomain(u), new double[]{0.3, 0.1, 0.2, 0.4 }));
        model.setFactor(x, EquationBuilder.of(model).fromVector(x, 0,0,0,1));

        BayesianNetwork emp = model.getEmpiricalNet();

        BayesianFactor Ptrue = emp.getFactor(x);

        System.out.println(Ptrue);



        System.out.println(model.toVCredal(BayesianFactor.combineAll(model.getEmpiricalNet().getFactors())));



    /*
    K(vars[0]|[]) [0.6, 0.0, 0.0, 0.4]
              [0.0, 0.0, 0.6, 0.4]
              [0.0, 0.6, 0.0, 0.4]
]
     */

        TIntIntMap[] data = model.getEmpiricalNet().samples(N);

/*
        // randomize P(U)
        double initPoints[][] =
                new double[][]{
                        new double[]{0.8, 0.05, 0.05, 0.1},
                        new double[]{0.7, 0.1, 0.5, 0.15},
                        new double[]{0.7, 0.1, 0.1, 0.1},
                        new double[]{0.01, 0.7, 0.09, 0.2},
                        new double[]{0.01, 0.08, 0.9, 0.01},
                        new double[]{0.65, 0.05, 0.2, 0.1},
                        new double[]{0.2, 0.2, 0.2, 0.4},
                        new double[]{0.1, 0.3, 0.2, 0.4},
                        new double[]{0.3, 0.1, 0.1, 0.5},
                        new double[]{0.1, 0.2, 0.1, 0.6},
        };*/

        double s = 0.05;
        double initPoints[][] = getInitPoints(s);

        ArrayList<double[]> trajectories = new ArrayList<>();

        int i = 1;

        for (double[] initPoint : initPoints) {
            StructuralCausalModel rmodel = model.copy();
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
            //System.out.println(postModel);


            //em.getIntermediateModels().stream().forEach(m -> System.out.println(m.getFactor(u)));

            int num_it = em.getIntermediateModels().size();

         /*   System.out.println(
                    Arrays.toString(
                            Doubles.concat(initPoint, em.getIntermediateModels().get(num_it - 1).getFactor(u).getData())
                    ));
*/

            trajectories.add(Doubles.concat(initPoint, em.getIntermediateModels().get(num_it - 1).getFactor(u).getData()));

            System.out.println(i+"/"+initPoints.length + "\t"+((i*100.0)/initPoints.length)+"%");
            i++;
        }

        new WriterCSV(trajectories.toArray(double[][]::new), "./trajectories"+String.valueOf(s).replace(".","_")+".csv")
                .setVarNames("s1","s2","s3","s4","e1","e2","e3","e4")
                .write();



    }



    public static boolean isInside(double[] p){
        double lbounds[] = {0.0, 0.0, 0.0, 0.4};
        double ubounds[] = {0.6, 0.6, 0.6, 0.4};

        for (int i = 0; i<p.length; i++){
            if(p[i]<lbounds[i] || p[i]>ubounds[i])
                return false;
        }
        return true;

    }

    public static double[][] getInitPoints(double s){

        ArrayList<double[]> points = new ArrayList();
        for(double p1=s; p1<=1; p1+=s) {
            for (double p2 = s; p2 <= 1 - p1; p2 += s) {
                for (double p3 = s; p3 <= 1 - (p1 + p2); p3 += s) {
                    for (double p4 = s; p4 <= 1 - (p1 + p2 + p3); p4 += s) {
                        double[] p = new double[]{p1,p2,p3,p4};
                        if(!isInside(p))
                            points.add(p);
                    }
                }

            }
        }
        return points.toArray(double[][]::new);
    }
}
