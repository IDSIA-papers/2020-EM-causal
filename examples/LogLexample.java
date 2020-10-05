import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.crema.data.WriterCSV;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import gnu.trove.map.TIntIntMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class LogLexample {
    public static void main(String[] args) throws IOException {

        int N = 1000;

        StructuralCausalModel model = new StructuralCausalModel();
        int u = model.addVariable(3, true);
        int x = model.addVariable(2);
        model.addParents(x, u);


        model.setFactor(u, new BayesianFactor(model.getDomain(u), new double[]{0.3, 0.1, 0.6 }));
        model.setFactor(x, EquationBuilder.of(model).fromVector(x, 0,0,1));

        BayesianNetwork emp = model.getEmpiricalNet();

        BayesianFactor Ptrue = emp.getFactor(x);

        TIntIntMap[] data =  emp.samples(N);


        List<double[]> llist = new ArrayList();

        for(double p1=0.0; p1<=1; p1+=0.01){
            for(double p2=0.0; p2<=1-p1; p2+=0.01){
                double p3 = 1 - (p1+p2);
                System.out.print(p1+","+p2+","+p3+",");

                double[] p = new double[]{p1,p2,p3};
                model.setFactor(u, new BayesianFactor(model.getDomain(u), p));
                double l = model.getEmpiricalNet().sumLogProb(data);

                System.out.println(l);

                llist.add(new double[]{p1,p2,p3,l});

            }
        }


        double[][] llvalues = llist.toArray(double[][]::new)
                ;
        new WriterCSV(llvalues, "./examples/llvalues.csv")
                .setVarNames("p1","p2","p3","ll")
                .write();


    }
}
