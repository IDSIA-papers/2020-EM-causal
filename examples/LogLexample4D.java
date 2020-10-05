import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.crema.data.WriterCSV;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import gnu.trove.map.TIntIntMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LogLexample4D {
    public static void main(String[] args) throws IOException {

        int N = 1000;

        StructuralCausalModel model = new StructuralCausalModel();
        int u = model.addVariable(4, true);
        int x = model.addVariable(2);
        model.addParents(x, u);


        model.setFactor(u, new BayesianFactor(model.getDomain(u), new double[]{0.3, 0.1, 0.2, 0.4 }));
        model.setFactor(x, EquationBuilder.of(model).fromVector(x, 0,0,0,1));

        BayesianNetwork emp = model.getEmpiricalNet();

        BayesianFactor Ptrue = emp.getFactor(x);

        TIntIntMap[] data =  emp.samples(N);


        model.toVCredal(Ptrue);



        List<double[]> llist = new ArrayList();

        for(double p1=0.01; p1<=1; p1+=0.025){
            for(double p2=0.01; p2<=1-p1; p2+=0.025){
                for(double p3=0.01; p3<=1-(p1+p2); p3+=0.025) {


                    double p4 = 1 - (p1 + p2 + p3);
                    if(p4!=0) {
                        System.out.print(p1 + "\t" + p2 + "\t" + p3 + "\t" + p4);

                        double[] p = new double[]{p1, p2, p3, p4};
                        model.setFactor(u, new BayesianFactor(model.getDomain(u), p));
                        double l = model.getEmpiricalNet().sumLogProb(data);

                        System.out.println("\t" + l + "\t" + (p1 + p2 + p3 + p4));

                        llist.add(new double[]{p1, p2, p3, p4, l});
                    }
                }

            }
        }


        double[][] llvalues = llist.toArray(double[][]::new)
                ;
        new WriterCSV(llvalues, "./examples/llvalues4D.csv")
                .setVarNames("p1","p2","p3", "p4","ll")
                .write();



    }
}
